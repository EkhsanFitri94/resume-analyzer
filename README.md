# 🎯 AI Resume Analyzer & Job Matcher

**Upload your resume + paste a job description → get instant match score, missing skills, and AI-powered suggestions.**

Built by [Ekhsan Fitri](https://github.com/EkhsanFitri94) — AI-powered career tool for job seekers.

---

## ✨ Features

- 📄 **Multi-format resume parsing** — PDF, DOCX, TXT support
- 🔍 **Skill extraction** — auto-detect technical, soft, and business skills
- 📊 **Match scoring** — weighted algorithm: skill match (60%) + text similarity (40%)
- ❌ **Gap detection** — missing skills ranked by importance
- 🤖 **ATS optimization** — keyword scanner for Applicant Tracking Systems
- 💡 **Actionable suggestions** — concrete tips to improve your resume
- 🧠 **AI enhancement** — optional OpenAI integration for deeper analysis
- 📈 **Visual breakdown** — score circle, skill tags, detailed report

## 🚀 Quick Start

```bash
git clone https://github.com/EkhsanFitri94/resume-analyzer.git
cd resume-analyzer
pip install -r requirements.txt
streamlit run app.py
```

## 🔑 AI Mode (Optional)

Enter an OpenAI API key in the app for GPT-enhanced analysis:
- Personalized resume coaching
- Role-specific positioning tips
- Bullet point rewrites

The app works fully without an API key — using TF-IDF + skill matching.

## 🎯 How It Works

1. **Upload resume** (PDF/DOCX/TXT) or paste text
2. **Paste job description**
3. **Click Analyze** → instant results:

| Tab | Content |
|---|---|
| 📊 **Skill Analysis** | Matching vs missing skills, score breakdown |
| 💡 **Suggestions** | Concrete improvement tips, ATS keywords |
| 🤖 **AI Insights** | GPT-powered personalized coaching |
| 📋 **Details** | Raw analysis data (JSON) |

## 🛠️ Tech Stack

- **Parsing:** PyPDF2, python-docx
- **NLP:** scikit-learn (TF-IDF + cosine similarity)
- **Analysis:** Custom skill database (100+ technical, soft, business skills)
- **AI:** OpenAI API (optional)
- **UI:** Streamlit

---

*Part of Ekhsan Fitri's AI & Analytics portfolio · [More projects](https://github.com/EkhsanFitri94)*
