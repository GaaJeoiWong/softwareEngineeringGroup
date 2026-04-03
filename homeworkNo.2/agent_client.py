import asyncio
import os
import json
from google import genai
from google.genai import types
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession

# ================= 配置区 =================
# 请确保你已经在环境变量中设置了 GOOGLE_API_KEY
# 如果没设置，可以临时取消下面这行的注释并填入（但不建议长期这样做）
# os.environ["GOOGLE_API_KEY"] = "你的_API_KEY"

MODEL_NAME = "models/gemini-2.5-flash" 

# 2. 初始化客户端
# 注意：2026 年的新 SDK 默认就走最新接口，http_options 也可以去掉了
client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

async def chat_loop():
    # 1. 配置如何启动 MCP Server
    server_params = StdioServerParameters(
        command="python",
        args=["mcp_server.py"], # 确保当前目录下有这个文件
    )

    print("🔄 正在连接 MCP 助理服务器...")
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # 初始化 MCP 会话
            await session.initialize()
            
            # 2. 获取 Server 工具并转换为 Google 原生格式 (Function Declarations)
            mcp_tools = await session.list_tools()
            google_tools = []
            
            for tool in mcp_tools.tools:
                google_tools.append(
                    types.FunctionDeclaration(
                        name=tool.name,
                        description=tool.description,
                        parameters=tool.inputSchema
                    )
                )
            
            # 封装进 Tool 对象
            tool_config = [types.Tool(function_declarations=google_tools)] if google_tools else None

            print("✅ Gemini 原生助理已就绪！(输入 'quit' 退出)")

            # 3. 创建原生对话会话 (Chat Session)
            # Google SDK 会自动维护上下文，无需手动管理 messages 列表
            chat = client.chats.create(
                model=MODEL_NAME,
                config=types.GenerateContentConfig(
                    system_instruction="你是一个全能的AI助手。你可以通过调用提供的工具来帮助用户查询、修改文件或执行任务。请保持专业且友好的态度。",
                    tools=tool_config
                )
            )

            while True:
                user_input = input("\n👨‍💻 用户: ")
                if user_input.lower() in ['quit', 'exit']:
                    break

                # 4. 发送用户消息
                # 使用 SDK 的循环逻辑处理可能的连续工具调用
                try:
                    response = chat.send_message(user_input)
                    
                    # 5. 核心逻辑：检查模型是否发出了“工具调用”请求
                    # Gemini 可能会连续多次调用工具，所以这里用循环处理
                    while True:
                        # 查找当前响应中是否有 function_call
                        call = None
                        for part in response.candidates[0].content.parts:
                            if part.function_call:
                                call = part.function_call
                                break
                        
                        if not call:
                            break # 如果没有工具调用了，退出循环，输出最终文字
                        
                        tool_name = call.name
                        tool_args = call.args
                        
                        print(f"   [🤖 助理正在操作: 调用 {tool_name}，参数: {tool_args}]")
                        
                        # 6. 通过 MCP 协议执行真实操作
                        try:
                            result = await session.call_tool(tool_name, tool_args)
                            # 提取结果文本
                            tool_result_text = result.content[0].text if result.content else "Done"
                        except Exception as e:
                            tool_result_text = f"执行出错: {str(e)}"
                            
                        print(f"   [⚙️ 操作完成: 反馈结果中...]")
                        
                        # 7. 将工具结果反馈给 Gemini，让它继续思考
                        response = chat.send_message(
                            types.Part.from_function_response(
                                name=tool_name,
                                response={'result': tool_result_text}
                            )
                        )

                    # 输出 Gemini 最终生成的自然语言回复
                    print(f"\n🤖 助理: {response.text}")

                except Exception as e:
                    print(f"\n❌ 发生错误: {str(e)}")

if __name__ == "__main__":
    # 运行异步循环
    try:
        asyncio.run(chat_loop())
    except KeyboardInterrupt:
        print("\n👋 已退出。")
