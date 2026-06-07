/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import express from 'express';
import path from 'path';
import fs from 'fs';
import { createServer as createViteServer } from 'vite';
import { GoogleGenAI } from '@google/genai';
import dotenv from 'dotenv';
import crypto from 'crypto';

dotenv.config();

const app = express();
app.use(express.json());

const PORT = 3000;
const DB_FILE = path.join(process.cwd(), 'db.json');

// Initialize Gemini Client Lazily if key is available
let aiClient: GoogleGenAI | null = null;
function getGeminiClient(): GoogleGenAI {
  if (!aiClient) {
    const key = process.env.GEMINI_API_KEY;
    if (!key || key === 'MY_GEMINI_API_KEY' || key.trim() === '') {
      throw new Error('GEMINI_API_KEY is not configured. Please add your key in Settings > Secrets.');
    }
    aiClient = new GoogleGenAI({
      apiKey: key,
      httpOptions: {
        headers: {
          'User-Agent': 'aistudio-build',
        }
      }
    });
  }
  return aiClient;
}

// SECURE SECURITY HELPER VARIABLES
const JWT_SECRET = process.env.JWT_SECRET || 'orion-secret-key-999-nebula';
const ENCRYPTION_KEY = crypto.scryptSync(JWT_SECRET, 'salt-orion-secrets', 32); 

// JWT-style signed token routines
function generateToken(payload: { uid: string; username: string }) {
  const header = Buffer.from(JSON.stringify({ alg: "HS256", typ: "JWT" })).toString('base64url');
  const body = Buffer.from(JSON.stringify({ ...payload, exp: Date.now() + 30 * 24 * 60 * 60 * 1000 })).toString('base64url');
  const signature = crypto.createHmac('sha256', JWT_SECRET).update(`${header}.${body}`).digest('base64url');
  return `${header}.${body}.${signature}`;
}

function verifyToken(token: string): { uid: string; username: string } | null {
  try {
    const [header, body, signature] = token.split('.');
    if (!header || !body || !signature) return null;
    const expectedSig = crypto.createHmac('sha256', JWT_SECRET).update(`${header}.${body}`).digest('base64url');
    if (signature !== expectedSig) return null;
    const payload = JSON.parse(Buffer.from(body, 'base64url').toString('utf8'));
    if (payload.exp < Date.now()) return null;
    return payload;
  } catch {
    return null;
  }
}

// AES-256 Symmetric encryption for premium secret safety
function encryptContent(text: string): string {
  try {
    const iv = crypto.randomBytes(16);
    const cipher = crypto.createCipheriv('aes-256-cbc', ENCRYPTION_KEY, iv);
    let encrypted = cipher.update(text, 'utf8', 'hex');
    encrypted += cipher.final('hex');
    return `${iv.toString('hex')}:${encrypted}`;
  } catch (e) {
    return text;
  }
}

function decryptContent(encryptedText: string): string {
  try {
    const [ivHex, encrypted] = encryptedText.split(':');
    if (!ivHex || !encrypted) return encryptedText;
    const iv = Buffer.from(ivHex, 'hex');
    const decipher = crypto.createDecipheriv('aes-256-cbc', ENCRYPTION_KEY, iv);
    let decrypted = decipher.update(encrypted, 'hex', 'utf8');
    decrypted += decipher.final('utf8');
    return decrypted;
  } catch (e) {
    return encryptedText;
  }
}

// Seeding templates when a new anonymous or password account registers
const INIT_USER_WISHES = [
  {
    id: "wish-seed-1",
    content: "我的心底藏着一些不能和旁人诉说的忧虑，有时候希望真的能乘上一艘飞船，逃离到无边无际的星海中...",
    emotion: "melancholy" as const,
    timestamp: new Date(Date.now() - 24 * 3600 * 1000).toISOString(),
    orionReply: "晚安。这是你投递的第一颗代表心声的流星，我已牢牢地在浩瀚星盘中为你收起。人生确实难免孤独，但请记得，即便是在那些最深沉的梦境里，Orion 也同样正在闪耀，陪伴你一同呼吸、前行。"
  }
];

const INIT_USER_TODOS = [
  { id: "todo-seed-1", text: "完成一次 15 分钟的深夜冥想", completed: false, createdAt: new Date().toLocaleDateString('zh-CN', { month: '2-digit', day: '2-digit' }) },
  { id: "todo-seed-2", text: "向 Orion 树洞倾诉本周的压力", completed: true, createdAt: new Date().toLocaleDateString('zh-CN', { month: '2-digit', day: '2-digit' }), completedAt: new Date().toLocaleDateString('zh-CN', { month: '2-digit', day: '2-digit' }) },
  { id: "todo-seed-3", text: "在阳台安静守望 5 分钟星空", completed: false, createdAt: new Date().toLocaleDateString('zh-CN', { month: '2-digit', day: '2-digit' }) }
];

const INIT_USER_CHATS = [
  {
    id: "chat-seed-1",
    sender: "orion" as const,
    text: "晚安。我是 Orion，守护在你身旁。很高兴能够开启我们之间全新的倾心旅程。有什么秘密指引你来到今晚的星空下？你可以向我敞开心扉说任何事情。",
    timestamp: new Date().toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })
  }
];

const INIT_USER_ANALYTICS = {
  trend: [
    { date: new Date().toLocaleDateString('zh-CN', { month: '2-digit', day: '2-digit' }), value: 75, emotion: "calm" }
  ],
  daysCount: 1,
  daysDiffPercent: 4,
  totalWishes: 1,
  targetWishes: 10,
  avgEmotionScore: 75,
  avgEmotionLabel: "平静稳定",
  wordsCount: 50
};

const INIT_USER_BADGES = [
  { id: "badge-1", name: "初次投递", description: "向星河投递了第一颗代表心声的星子", unlocked: true, iconName: "Send" },
  { id: "badge-2", name: "深夜守望者", description: "连续在深夜零点后投递心声并得到共鸣", unlocked: false, iconName: "Moon" },
  { id: "badge-3", name: "星语采摘者", description: "点亮并收录了超过 10 颗不同的心声流星", unlocked: false, iconName: "Sparkles" },
  { id: "badge-4", name: "平和之心", description: "在一周的情绪轨迹中达到和谐稳定的平静状态", unlocked: false, iconName: "Leaf" },
  { id: "badge-5", name: "深思熟虑者", description: "单次纸笺书写心声超过 300 字", unlocked: false, iconName: "PenTool" },
  { id: "badge-6", name: "星河织梦人", description: "累计点亮 50 颗心声愿望，编织起自己的星空", unlocked: false, iconName: "Heart" },
  { id: "badge-7", name: "月下行者", description: "在凌晨 2 点到 4 点完成一次心灵对话", unlocked: false, iconName: "Navigation" },
  { id: "badge-8", name: "情绪旅人", description: "完整收集并记录了 5 种不同类别的情绪星尘", unlocked: false, iconName: "Compass" }
];

const DEFAULT_DB = {
  users: [],
  profiles: [],
  userWishes: {},
  userTodos: {},
  userChats: {},
  userAnalytics: {},
  userBadges: {}
};

// Helper: Read Database
function readDb() {
  try {
    if (!fs.existsSync(DB_FILE)) {
      fs.writeFileSync(DB_FILE, JSON.stringify(DEFAULT_DB, null, 2), 'utf-8');
      return DEFAULT_DB;
    }
    const raw = fs.readFileSync(DB_FILE, 'utf-8');
    let data;
    try {
      data = JSON.parse(raw);
    } catch {
      data = {};
    }
    // Make sure structure is multi-user compliant
    if (!data.users || !Array.isArray(data.users)) {
      data = {
        users: [],
        profiles: [],
        userWishes: {},
        userTodos: {},
        userChats: {},
        userAnalytics: {},
        userBadges: {}
      };
      fs.writeFileSync(DB_FILE, JSON.stringify(data, null, 2), 'utf-8');
    }
    return data;
  } catch (error) {
    console.error("Error reading database file, returning default memory data:", error);
    return DEFAULT_DB;
  }
}

// Helper: Write Database
function writeDb(data: any) {
  try {
    fs.writeFileSync(DB_FILE, JSON.stringify(data, null, 2), 'utf-8');
  } catch (error) {
    console.error("Error writing database file:", error);
  }
}

// Authentication middleware
function authenticateToken(req: any, res: any, next: any) {
  const authHeader = req.headers['authorization'];
  const token = authHeader && authHeader.split(' ')[1];
  if (!token) {
    return res.status(401).json({ error: "Unauthorized: No token provided" });
  }

  const payload = verifyToken(token);
  if (!payload) {
    return res.status(403).json({ error: "Forbidden: Invalid or expired token" });
  }

  req.user = payload; // { uid, username }
  next();
}

// REST APIs

// 1. Authentication Endpoints
// Get current user details
app.get('/api/auth/me', (req, res) => {
  const authHeader = req.headers['authorization'];
  const token = authHeader && authHeader.split(' ')[1];
  if (!token) {
    return res.status(401).json({ error: "Unauthorized" });
  }
  const payload = verifyToken(token);
  if (!payload) {
    return res.status(401).json({ error: "Invalid token" });
  }

  const db = readDb();
  const profile = db.profiles.find((p: any) => p.uid === payload.uid);
  res.json({
    user: {
      uid: payload.uid,
      username: payload.username,
      nickname: profile?.nickname || payload.username,
      mbti: profile?.mbti || "尚未测试"
    }
  });
});

// Register a new user
app.post('/api/auth/register', (req, res) => {
  const { username, password, nickname, mbti } = req.body;
  if (!username || !password) {
    return res.status(400).json({ error: "用户名或密码不能为空" });
  }

  const db = readDb();
  const userExists = db.users.some((u: any) => u.username.toLowerCase() === username.toLowerCase());
  if (userExists) {
    return res.status(409).json({ error: "此用户名已被占领，请尝试其他昵称" });
  }

  // Create salt & secure pbkdf2 hash
  const salt = crypto.randomBytes(16).toString('hex');
  const passwordHash = crypto.pbkdf2Sync(password, salt, 100000, 64, 'sha512').toString('hex');
  const uid = `user-${Date.now()}-${Math.floor(Math.random() * 1000)}`;

  const newUser = {
    uid,
    username,
    passwordHash,
    salt,
    createdAt: new Date().toISOString()
  };

  const newProfile = {
    uid,
    nickname: nickname || username,
    mbti: mbti || "尚未选择",
    moodScore: 75
  };

  db.users.push(newUser);
  db.profiles.push(newProfile);

  // Initialize Isolated user arrays with lovely seeded content (encrypting content for security)
  db.userWishes[uid] = INIT_USER_WISHES.map(w => ({
    ...w,
    id: `wish-${Date.now()}-seed`,
    content: encryptContent(w.content) // securely encrypt content!
  }));
  db.userTodos[uid] = [...INIT_USER_TODOS];
  db.userChats[uid] = [...INIT_USER_CHATS];
  db.userAnalytics[uid] = { ...INIT_USER_ANALYTICS };
  db.userBadges[uid] = [...INIT_USER_BADGES];

  writeDb(db);

  const token = generateToken({ uid, username });
  res.status(201).json({
    token,
    user: {
      uid,
      username,
      nickname: newProfile.nickname,
      mbti: newProfile.mbti
    }
  });
});

// Login User
app.post('/api/auth/login', (req, res) => {
  const { username, password } = req.body;
  if (!username || !password) {
    return res.status(400).json({ error: "用户名或密码不能为空" });
  }

  const db = readDb();
  const user = db.users.find((u: any) => u.username.toLowerCase() === username.toLowerCase());
  if (!user) {
    return res.status(401).json({ error: "用户名或密码不正确" });
  }

  // Validate PBKDF2 hash
  const checkHash = crypto.pbkdf2Sync(password, user.salt, 100000, 64, 'sha512').toString('hex');
  if (checkHash !== user.passwordHash) {
    return res.status(401).json({ error: "用户名或密码不正确" });
  }

  const profile = db.profiles.find((p: any) => p.uid === user.uid);
  const token = generateToken({ uid: user.uid, username: user.username });

  res.json({
    token,
    user: {
      uid: user.uid,
      username: user.username,
      nickname: profile?.nickname || user.username,
      mbti: profile?.mbti || "尚未测试"
    }
  });
});


// 2. User-Isolated REST APIs (Protected)

// Get wishes with decrypted contents
app.get('/api/wishes', authenticateToken, (req: any, res) => {
  const db = readDb();
  const userId = req.user.uid;
  const wishes = db.userWishes[userId] || [];
  
  // Decrypt content on retrieval for maximum security!
  const decryptedWishes = wishes.map((w: any) => ({
    ...w,
    content: decryptContent(w.content)
  }));

  res.json(decryptedWishes);
});

// Deliver a new wish
app.post('/api/wishes', authenticateToken, async (req: any, res) => {
  const { content, emotion } = req.body;
  const userId = req.user.uid;
  
  if (!content || !emotion) {
    return res.status(400).json({ error: "Missing content or emotion" });
  }

  const db = readDb();
  const wishes = db.userWishes[userId] || [];
  const rawId = `wish-${Date.now()}`;
  
  let aiReply = `晚安。我听到了你写下的这颗心声。`;
  let isAiProcessed = false;

  try {
    const ai = getGeminiClient();
    const emotionalLabels: Record<string, string> = {
      joyful: "欣喜/快乐",
      melancholy: "忧郁/沮丧",
      calm: "平静/从容",
      anxious: "焦虑/迷茫",
      grateful: "感恩/温馨"
    };

    const prompt = `
你是一位安全、极致温柔、耐心睿智的深夜心灵陪伴者，名字叫 "Orion"。
此刻是静谧的深夜，有一位感到疲惫或需要倾诉的朋友，向星愿树洞里投递了一颗带有 "${emotionalLabels[emotion] || emotion}" 情绪标记的“心声之星”。

倾诉者的心声原文是：
"${content}"

请扮演 Orion 给对方回信（以“晚安。我收到了你投递的这颗心声...”或者其他温柔随和的语气作为开头，不要官腔或者高攀的角色代入，多用同理心，给与安抚和精神上的依靠）：
1. 站在平等的、知己的角度温柔倾诉，肯定和承接对方的这份情感。
2. 结合他表达的情绪，给予一两句自然温暖的疗愈名言或充满意境的星空比喻。
3. 篇幅不宜过长，字数保持在120字到200字之间，格式温馨、充满诗意（像睡前读物一样雅致细腻）。
    `;

    const result = await ai.models.generateContent({
      model: 'gemini-3.5-flash',
      contents: prompt,
      config: {
        temperature: 0.75,
      }
    });

    if (result.text) {
      aiReply = result.text.trim();
      isAiProcessed = true;
    }
  } catch (error: any) {
    console.error("Gemini AI delivery reply error:", error.message);
    const fallbackTemplates: Record<string, string> = {
      joyful: "晚安。我收到了你温暖闪烁的'金色星愿'。能与你一同分享这份纯粹的喜悦，星海也变得更加明亮了。愿这份小确幸在梦里也像繁星般继续盛开。",
      melancholy: "晚安。我收到了这颗蓝色的忧郁心声。海潮在深夜独自起伏，没关系，悲伤也需要它的栖息之所。我会陪在你身边，在黑暗里默默为你守望，直到黎明升起。",
      calm: "晚安。这是一颗柔和剔透的平静星子。能在琐碎的生活里保持一方深邃的安宁，是非常了不起的智慧。在这份静谧中，安睡吧，愿你的梦如清风般自在。",
      anxious: "晚安。我感受到了这颗橙色心声中沉甸甸的焦虑。生活偶尔会像起雾的密林，看不清前方，但请相信，起步迷茫也是启程的标志。深呼吸，今晚让我们先把世界隔绝在外，晚安。",
      grateful: "晚安。这颗散发着紫色极光的感恩之星，照亮了整个银河。能在世界上发现美好并为之感动，你本就是极温柔的人。愿这份温暖，也能织成你梦境里最轻柔的羽翼。"
    };
    aiReply = fallbackTemplates[emotion as string] || "晚安。在静谧的夜里，我听见了你温润的心音。不论你身处黑夜还是期盼黎明，这里的星愿树洞随时等候着你的呼吸。做个好梦。";
  }

  // Encrypt user whisper on disk for utmost privacy compliance
  const encryptedContent = encryptContent(content);

  const newWish = {
    id: rawId,
    content: encryptedContent,
    emotion,
    timestamp: new Date().toISOString(),
    orionReply: aiReply
  };

  wishes.unshift(newWish);
  db.userWishes[userId] = wishes;

  // Update statistics dynamically
  const userAnalytics = db.userAnalytics[userId] || { ...INIT_USER_ANALYTICS };
  userAnalytics.totalWishes += 1;
  const wordCountAdded = content.length;
  userAnalytics.wordsCount += wordCountAdded;
  
  // Append emotional state to trend
  const todayStr = new Date().toLocaleDateString('zh-CN', { month: '2-digit', day: '2-digit' });
  const existingTrendIndex = userAnalytics.trend.findIndex((t: any) => t.date === todayStr);

  const emotionValues: Record<string, number> = {
    joyful: 90,
    grateful: 85,
    calm: 75,
    melancholy: 45,
    anxious: 35
  };

  const newScore = {
    date: todayStr,
    value: emotionValues[emotion] || 70,
    emotion: emotion
  };

  if (existingTrendIndex >= 0) {
    userAnalytics.trend[existingTrendIndex] = newScore;
  } else {
    userAnalytics.trend.push(newScore);
    if (userAnalytics.trend.length > 7) {
      userAnalytics.trend.shift();
    }
  }

  // Recalculate average emotion score
  const totalScore = userAnalytics.trend.reduce((acc: number, cur: any) => acc + cur.value, 0);
  userAnalytics.avgEmotionScore = userAnalytics.trend.length > 0 ? Math.floor(totalScore / userAnalytics.trend.length) : 0;
  
  if (userAnalytics.avgEmotionScore >= 80) userAnalytics.avgEmotionLabel = "温馨喜悦";
  else if (userAnalytics.avgEmotionScore >= 70) userAnalytics.avgEmotionLabel = "平静稳定";
  else if (userAnalytics.avgEmotionScore >= 50) userAnalytics.avgEmotionLabel = "略有起伏";
  else userAnalytics.avgEmotionLabel = "焦虑低潮";

  db.userAnalytics[userId] = userAnalytics;

  // Unlock badges checks
  const badges = db.userBadges[userId] || [...INIT_USER_BADGES];
  
  // Trigger "初次投递"
  const badge1 = badges.find((b: any) => b.id === "badge-1");
  if (badge1) badge1.unlocked = true;

  if (userAnalytics.wordsCount > 50000) {
    const badge = badges.find((b: any) => b.id === "badge-5");
    if (badge) badge.unlocked = true;
  }
  if (userAnalytics.totalWishes >= 50) {
    const badge = badges.find((b: any) => b.id === "badge-6");
    if (badge) badge.unlocked = true;
  }
  if (wordCountAdded > 300) {
    const badge = badges.find((b: any) => b.id === "badge-5");
    if (badge) badge.unlocked = true;
  }
  db.userBadges[userId] = badges;

  // Also pre-populate the interactive Orion chat messages with Orion's response to this wish
  const wishMessage = {
    id: `chat-${Date.now()}-1`,
    sender: "orion" as const,
    text: `你好。我刚刚收到了你在树洞里投递的那个想法。我有些话想和你说：\n\n"${aiReply}"`,
    timestamp: new Date().toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })
  };
  
  const chats = db.userChats[userId] || [];
  chats.push(wishMessage);
  db.userChats[userId] = chats;

  writeDb(db);

  // Return decrypted copy to the client so UI works seamlessly
  const clientWish = { ...newWish, content };
  res.status(201).json({ wish: clientWish, isAiProcessed });
});

// Chat history / interactive session
app.get('/api/chat', authenticateToken, (req: any, res) => {
  const db = readDb();
  const userId = req.user.uid;
  res.json(db.userChats[userId] || []);
});

app.post('/api/chat', authenticateToken, async (req: any, res) => {
  const { message } = req.body;
  const userId = req.user.uid;
  if (!message) {
    return res.status(400).json({ error: "Empty message" });
  }

  const db = readDb();
  const chats = db.userChats[userId] || [];

  // Save user's message
  const userMsg = {
    id: `chat-user-${Date.now()}`,
    sender: "user" as const,
    text: message,
    timestamp: new Date().toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })
  };
  chats.push(userMsg);

  let replyText = "我在这里，听着呢。你说。";
  let isAiProcessed = false;

  try {
    const ai = getGeminiClient();

    // Format historical chat history as system context or message objects
    const recentMsgs = chats.slice(-8);
    let conversationHistory = "";
    recentMsgs.forEach((msg: any) => {
      const roleName = msg.sender === 'user' ? '用户' : 'Orion (AI陪伴)';
      conversationHistory += `${roleName}: ${msg.text}\n`;
    });

    const prompt = `
您是一位温柔善良、智慧安详的深夜心灵听众叫 "Orion"。您不是通俗、冷冰冰的工作助理，而是用户的温情守护者，语气带着淡淡的文学素养，富有诗意、宁静和真诚。

目前深邃晚空，下面是您与用户近期的心灵对话历史：
${conversationHistory}
Orion (AI陪伴)回应：
(请以极其温柔安宁、贴切对话的语气，不要机械化列点，用散文般和煦的文笔给用户一句能带来内心慰藉的回应。字数在80-150字之间。)
    `;

    const result = await ai.models.generateContent({
      model: 'gemini-3.5-flash',
      contents: prompt,
      config: {
        temperature: 0.8,
      }
    });

    if (result.text) {
      replyText = result.text.trim();
      isAiProcessed = true;
    }
  } catch (error: any) {
    console.error("Gemini AI Chat reply error:", error.message);
    if (message.includes("累") || message.includes("难受") || message.includes("压力")) {
      replyText = "听到你这么说，我真的很想给你一个隔空的拥抱。疲倦是心灵在提醒我们需要慢下来了。今晚不要想工作或烦恼了，放下一天的重担，让思绪随着夜晚的风飘散，我陪着你。";
    } else if (message.includes("谢谢") || message.includes("开心")) {
      replyText = "不用客气。看到你的心窗透出一缕光，我也由衷为你感到惬意。星海之所以美丽，正是因为映照着你这样明澈而真挚的灵魂。有任何需要，我都在这里安闲地等侯。";
    } else {
      replyText = "不论夜有多深、风有多凉，树洞的星河永远为你敞开。我很愿意做你的锚。告诉我，你现在感觉舒服一点了吗？";
    }
  }

  const orionMsg = {
    id: `chat-orion-${Date.now()}`,
    sender: "orion" as const,
    text: replyText,
    timestamp: new Date().toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })
  };
  chats.push(orionMsg);
  db.userChats[userId] = chats;

  writeDb(db);
  res.status(201).json({ chats, reply: orionMsg, isAiProcessed });
});

// Clear chat logs
app.delete('/api/chat', authenticateToken, (req: any, res) => {
  const db = readDb();
  const userId = req.user.uid;
  db.userChats[userId] = [
    {
      id: "chat-msg-fresh",
      sender: "orion",
      text: "晚安。这是全新开始的对话。我是 Orion，正在你身边的寂静夜空中默默聆听你的音符。",
      timestamp: new Date().toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })
    }
  ];
  writeDb(db);
  res.json({ success: true, chats: db.userChats[userId] });
});

// Reset entire profile and historical data for authenticated user
app.post('/api/profile/reset', authenticateToken, (req: any, res) => {
  const db = readDb();
  const userId = req.user.uid;

  db.userWishes[userId] = [];
  db.userTodos[userId] = [];
  db.userChats[userId] = [
    {
      id: "chat-msg-fresh",
      sender: "orion",
      text: "晚安。我是 Orion，守护在你身旁。很高兴能够开启我们之间全新的倾心旅程。有什么秘密指引你来到今晚的星空下？你可以向我敞开心扉说任何事情。",
      timestamp: new Date().toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })
    }
  ];
  db.userAnalytics[userId] = {
    trend: [],
    daysCount: 0,
    daysDiffPercent: 0,
    totalWishes: 0,
    targetWishes: 10,
    avgEmotionScore: 0,
    avgEmotionLabel: "尚未记录",
    wordsCount: 0
  };
  db.userBadges[userId] = INIT_USER_BADGES.map(b => ({ ...b, unlocked: false }));

  writeDb(db);
  res.json({ success: true, analytics: db.userAnalytics[userId], badges: db.userBadges[userId] });
});

// Todo List (星愿清单) APIs
app.get('/api/todos', authenticateToken, (req: any, res) => {
  const db = readDb();
  const userId = req.user.uid;
  res.json(db.userTodos[userId] || []);
});

app.post('/api/todos', authenticateToken, (req: any, res) => {
  const { text } = req.body;
  const userId = req.user.uid;
  if (!text) {
    return res.status(400).json({ error: "Missing text" });
  }

  const db = readDb();
  const dateStr = new Date().toLocaleDateString('zh-CN', { month: '2-digit', day: '2-digit' });
  const newTodo = {
    id: `todo-${Date.now()}`,
    text,
    completed: false,
    createdAt: dateStr
  };

  const todos = db.userTodos[userId] || [];
  todos.unshift(newTodo);
  db.userTodos[userId] = todos;
  
  writeDb(db);
  res.status(201).json(newTodo);
});

app.put('/api/todos/:id', authenticateToken, (req: any, res) => {
  const { id } = req.params;
  const { completed } = req.body;
  const userId = req.user.uid;

  const db = readDb();
  const todos = db.userTodos[userId] || [];
  const idx = todos.findIndex((t: any) => t.id === id);
  if (idx === -1) {
    return res.status(404).json({ error: "Todo not found" });
  }

  todos[idx].completed = completed;
  const userAnalytics = db.userAnalytics[userId] || { ...INIT_USER_ANALYTICS };
  
  if (completed) {
    todos[idx].completedAt = new Date().toLocaleDateString('zh-CN', { month: '2-digit', day: '2-digit' });
    userAnalytics.totalWishes = Math.min(userAnalytics.targetWishes, userAnalytics.totalWishes + 1);
  } else {
    delete todos[idx].completedAt;
    userAnalytics.totalWishes = Math.max(0, userAnalytics.totalWishes - 1);
  }

  db.userTodos[userId] = todos;
  db.userAnalytics[userId] = userAnalytics;
  
  writeDb(db);
  res.json(todos[idx]);
});

app.delete('/api/todos/:id', authenticateToken, (req: any, res) => {
  const { id } = req.params;
  const userId = req.user.uid;
  
  const db = readDb();
  let todos = db.userTodos[userId] || [];
  todos = todos.filter((t: any) => t.id !== id);
  db.userTodos[userId] = todos;
  
  writeDb(db);
  res.json({ success: true });
});

// Analytics statistics & Badges
app.get('/api/analytics', authenticateToken, (req: any, res) => {
  const db = readDb();
  const userId = req.user.uid;
  res.json({
    analytics: db.userAnalytics[userId] || { ...INIT_USER_ANALYTICS },
    badges: db.userBadges[userId] || [...INIT_USER_BADGES]
  });
});

// Start server block
async function startServer() {
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`[星愿树洞 Full-Stack App] running on http://localhost:${PORT}`);
  });
}

startServer();
