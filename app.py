"""
AI Resume Analyzer & Job Matcher
Upload resume + job description → get match score, missing skills, and AI-powered suggestions.
"""
import streamlit as st
import re
import os
import json
from collections import Counter, defaultdict
from io import BytesIO
import math

# Optional imports with fallback
try:
    from PyPDF2 import PdfReader
    HAS_PDF = True
except ImportError:
    HAS_PDF = False

try:
    from docx import Document
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ── Skill Database ───────────────────────────────────────
TECH_SKILLS = {
    'python', 'sql', 'java', 'javascript', 'typescript', 'r', 'c++', 'c#', 'go', 'rust',
    'react', 'angular', 'vue', 'node.js', 'django', 'flask', 'fastapi', 'spring',
    'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform', 'jenkins', 'ci/cd',
    'tensorflow', 'pytorch', 'scikit-learn', 'pandas', 'numpy', 'spark', 'hadoop',
    'tableau', 'power bi', 'excel', 'looker', 'd3.js', 'plotly',
    'git', 'linux', 'bash', 'rest api', 'graphql', 'microservices',
    'machine learning', 'deep learning', 'nlp', 'computer vision', 'data science',
    'mongodb', 'postgresql', 'mysql', 'redis', 'elasticsearch', 'snowflake',
    'agile', 'scrum', 'jira', 'confluence', 'figma',
    'streamlit', 'beautiful soup', 'selenium', 'openai', 'langchain',
}

SOFT_SKILLS = {
    'communication', 'leadership', 'teamwork', 'problem solving', 'critical thinking',
    'time management', 'adaptability', 'creativity', 'collaboration', 'presentation',
    'project management', 'stakeholder management', 'negotiation', 'mentoring',
    'analytical', 'detail-oriented', 'self-motivated', 'strategic thinking',
    'conflict resolution', 'decision making', 'emotional intelligence',
}

BUSINESS_SKILLS = {
    'procurement', 'supply chain', 'vendor management', 'inventory management',
    'financial analysis', 'budgeting', 'forecasting', 'risk management',
    'business analysis', 'process improvement', 'operations management',
    'strategic planning', 'market research', 'competitive analysis',
    'contract negotiation', 'purchase order', 'invoice processing',
    'cost reduction', 'sourcing', 'category management',
}


# ── Resume Parsing ───────────────────────────────────────
def parse_resume(file) -> tuple:
    """Extract text from uploaded resume file (PDF, DOCX, or TXT)."""
    filename = file.name.lower()
    text = ""

    if filename.endswith('.pdf') and HAS_PDF:
        reader = PdfReader(BytesIO(file.read()))
        text = ' '.join(page.extract_text() or '' for page in reader.pages)
    elif filename.endswith('.docx') and HAS_DOCX:
        doc = Document(BytesIO(file.read()))
        text = ' '.join(p.text for p in doc.paragraphs)
    elif filename.endswith('.txt'):
        text = file.read().decode('utf-8', errors='ignore')
    else:
        text = file.read().decode('utf-8', errors='ignore')

    if not text.strip():
        raise ValueError("Could not extract text from file. Try TXT format.")

    return filename, text.strip()


# ── Skill Extraction ─────────────────────────────────────
def extract_skills(text: str) -> dict:
    """Extract technical, soft, and business skills from text."""
    text_lower = text.lower()[:8000]  # Limit for performance
    found_tech = set()
    found_soft = set()
    found_business = set()

    for skill in TECH_SKILLS:
        pattern = re.escape(skill.lower())
        if re.search(r'\b' + pattern + r'\b', text_lower):
            found_tech.add(skill.title())

    for skill in SOFT_SKILLS:
        pattern = re.escape(skill.lower())
        if re.search(r'\b' + pattern + r'\b', text_lower):
            found_soft.add(skill.title())

    for skill in BUSINESS_SKILLS:
        if skill.lower() in text_lower:
            found_business.add(skill.title())

    return {
        'technical': sorted(found_tech),
        'soft': sorted(found_soft),
        'business': sorted(found_business),
    }


# ── Match Analysis ───────────────────────────────────────
def analyze_match(resume_text: str, job_text: str) -> dict:
    """Compare resume against job description and compute match score."""
    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_text)

    # Find matching and missing skills
    all_resume = set(resume_skills['technical'] + resume_skills['soft'] + resume_skills['business'])
    all_job = set(job_skills['technical'] + job_skills['soft'] + job_skills['business'])

    matching = all_resume & all_job
    missing = all_job - all_resume
    extra = all_resume - all_job

    # Weighted scoring
    tech_match, tech_total = _category_score(resume_skills['technical'], job_skills['technical'])
    soft_match, soft_total = _category_score(resume_skills['soft'], job_skills['soft'])
    bus_match, bus_total = _category_score(resume_skills['business'], job_skills['business'])

    total = tech_total + soft_total + bus_total
    matched = tech_match + soft_match + bus_match

    if total == 0:
        skill_score = 0
    else:
        # Technical skills weighted 50%, soft 25%, business 25%
        tech_w = (tech_match / tech_total * 0.5) if tech_total > 0 else 0
        soft_w = (soft_match / soft_total * 0.25) if soft_total > 0 else 0
        bus_w = (bus_match / bus_total * 0.25) if bus_total > 0 else 0
        skill_score = round((tech_w + soft_w + bus_w) * 100)

    # TF-IDF cosine similarity (if sklearn available)
    tfidf_score = 0
    if HAS_SKLEARN and len(resume_text) > 50 and len(job_text) > 50:
        try:
            vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
            tfidf = vectorizer.fit_transform([resume_text[:5000], job_text[:5000]])
            tfidf_score = round(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0] * 100)
        except Exception:
            pass

    # Overall score: 60% skill match + 40% TF-IDF
    if tfidf_score > 0:
        overall = round(skill_score * 0.6 + tfidf_score * 0.4)
    else:
        overall = skill_score

    # Missing skills by importance
    missing_tech = set(job_skills['technical']) - set(resume_skills['technical'])
    missing_soft = set(job_skills['soft']) - set(resume_skills['soft'])
    missing_biz = set(job_skills['business']) - set(resume_skills['business'])

    # ATS keywords check
    ats_keywords = _extract_ats_keywords(job_text)
    ats_found = [kw for kw in ats_keywords if kw.lower() in resume_text.lower()]
    ats_missing = [kw for kw in ats_keywords if kw.lower() not in resume_text.lower()]

    # Suggestions
    suggestions = _generate_suggestions(missing_tech, missing_soft, missing_biz, ats_missing, overall)

    return {
        'overall_score': overall,
        'skill_score': skill_score,
        'tfidf_score': tfidf_score,
        'matching_skills': sorted(matching),
        'missing_tech': sorted(missing_tech),
        'missing_soft': sorted(missing_soft),
        'missing_biz': sorted(missing_biz),
        'extra_skills': sorted(extra),
        'ats_keywords_found': ats_found,
        'ats_keywords_missing': ats_missing,
        'suggestions': suggestions,
        'resume_skills': resume_skills,
        'job_skills': job_skills,
        'word_count': len(resume_text.split()),
    }


def _category_score(resume_skills: list, job_skills: list) -> tuple:
    """Calculate match score for a skill category."""
    job_set = set(job_skills)
    resume_set = set(resume_skills)
    if not job_set:
        return 0, 0
    matched = len(job_set & resume_set)
    return matched, len(job_set)


def _extract_ats_keywords(job_text: str) -> list:
    """Extract important keywords that ATS systems look for."""
    text_lower = job_text.lower()
    keywords = set()

    # Years of experience
    exp = re.findall(r'(\d+)[\+]?\s*(?:years|yrs)', text_lower)
    if exp:
        keywords.add(f"{exp[0]}+ years experience")

    # Education
    if 'bachelor' in text_lower or "bachelor's" in text_lower:
        keywords.add("Bachelor's Degree")
    if 'master' in text_lower or "master's" in text_lower:
        keywords.add("Master's Degree")
    if 'phd' in text_lower or 'doctorate' in text_lower:
        keywords.add("PhD/Doctorate")

    # Certifications
    certs = ['pmp', 'cpsm', 'cips', 'six sigma', 'scrum', 'aws certified', 'azure certified',
             'google certified', 'tableau certified', 'power bi certified']
    for cert in certs:
        if cert in text_lower:
            keywords.add(cert.upper())

    # Key phrases
    phrases = ['stakeholder management', 'data-driven', 'cross-functional',
               'end-to-end', 'process improvement', 'kpi', 'sla']
    for p in phrases:
        if p in text_lower:
            keywords.add(p.title())

    # Tools
    tools = ['excel', 'power bi', 'tableau', 'sap', 'oracle', 'salesforce', 'jira']
    for t in tools:
        if t in text_lower:
            keywords.add(t.upper() if t.isupper() else t.title())

    return sorted(keywords)


def _generate_suggestions(missing_tech, missing_soft, missing_biz, ats_missing, score) -> list:
    """Generate actionable resume improvement suggestions."""
    suggestions = []

    if score >= 80:
        suggestions.append({
            'icon': '✅',
            'priority': 'high',
            'title': 'Strong Match!',
            'detail': 'Your resume aligns well with this job. Focus on quantifying achievements and preparing for interviews.',
        })
    elif score >= 60:
        suggestions.append({
            'icon': '📝',
            'priority': 'medium',
            'title': 'Good Fit — minor improvements needed',
            'detail': 'Your profile matches many requirements. Add the missing keywords below to boost your ATS score.',
        })
    else:
        suggestions.append({
            'icon': '⚠️',
            'priority': 'high',
            'title': 'Significant gaps detected',
            'detail': 'Consider gaining experience in the missing skills or tailoring your resume to highlight transferable skills.',
        })

    # Missing tech skills
    top_missing = list(missing_tech)[:5]
    if top_missing:
        suggestions.append({
            'icon': '🔧',
            'priority': 'high',
            'title': f'Add these tech skills: {", ".join(top_missing)}',
            'detail': 'These are explicitly required. If you have experience, make sure they are clearly listed. If not, consider learning them.',
        })

    # Missing soft skills
    top_soft = list(missing_soft)[:3]
    if top_soft:
        suggestions.append({
            'icon': '💬',
            'priority': 'medium',
            'title': f'Highlight soft skills: {", ".join(top_soft)}',
            'detail': 'Weave these into your experience bullet points naturally, not just in a skills list.',
        })

    # ATS optimization
    if ats_missing:
        suggestions.append({
            'icon': '🤖',
            'priority': 'high',
            'title': f'ATS keywords missing: {", ".join(ats_missing[:5])}',
            'detail': 'ATS systems scan for exact keyword matches. Add these terms verbatim where applicable.',
        })

    # General
    suggestions.append({
        'icon': '📊',
        'priority': 'medium',
        'title': 'Quantify your impact',
        'detail': 'Replace generic descriptions with numbers: "Reduced costs by 15%" > "Helped reduce costs". Use metrics everywhere.',
    })

    suggestions.append({
        'icon': '📄',
        'priority': 'medium',
        'title': 'Keep it ATS-friendly',
        'detail': 'Use standard section headers (Experience, Education, Skills). Avoid tables, images, and complex formatting.',
    })

    return suggestions


# ── AI Enhancement ───────────────────────────────────────
def ai_enhance(resume_text: str, job_text: str, api_key: str = None) -> str:
    """Use AI to provide deeper analysis and personalized suggestions."""
    if not api_key or not api_key.startswith('sk-'):
        return None

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        prompt = f"""You are a professional resume coach and recruiter. Analyze this resume against the job description.
Provide a concise analysis (under 250 words) covering:
1. Overall match assessment
2. Top 3 things to improve in the resume
3. Best way to position the candidate for this role
4. One specific bullet point suggestion

Resume:
{resume_text[:2000]}

Job Description:
{job_text[:2000]}"""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=400,
        )
        return response.choices[0].message.content
    except Exception:
        return None


# ── Streamlit App ────────────────────────────────────────
st.set_page_config(page_title="AI Resume Analyzer", page_icon="🎯", layout="wide")

st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0c1929 0%, #1a2744 100%); }
    .score-circle {
        width: 180px; height: 180px; border-radius: 50%;
        background: conic-gradient(#4ade80 {score}%, #1e293b 0%);
        display: flex; align-items: center; justify-content: center;
        margin: 0 auto;
    }
    .score-inner {
        width: 140px; height: 140px; border-radius: 50%;
        background: #0f172a;
        display: flex; flex-direction: column;
        align-items: center; justify-content: center;
    }
    .skill-tag {
        display: inline-block; padding: 4px 10px; margin: 3px;
        border-radius: 20px; font-size: 0.8rem; font-weight: 600;
    }
    .skill-tag.match { background: rgba(74,222,128,0.15); color: #4ade80; border: 1px solid rgba(74,222,128,0.3); }
    .skill-tag.missing { background: rgba(248,113,113,0.15); color: #f87171; border: 1px solid rgba(248,113,113,0.3); }
    .skill-tag.extra { background: rgba(96,165,250,0.15); color: #60a5fa; border: 1px solid rgba(96,165,250,0.3); }
    .suggestion-box {
        background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px; padding: 1rem; margin-bottom: 0.7rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎯 AI Resume Analyzer & Job Matcher")
st.markdown("<p style='color:#94a3b8;'>Upload your resume, paste a job description, get instant match analysis</p>", unsafe_allow_html=True)

# ── Input ────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📄 Upload Resume")
    resume_file = st.file_uploader("PDF, DOCX, or TXT", type=['pdf', 'docx', 'txt'])
    resume_text_manual = st.text_area(
        "Or paste resume text",
        height=200,
        placeholder="Paste your resume content here...",
    )

with col2:
    st.markdown("### 📋 Job Description")
    job_text = st.text_area(
        "Paste job description",
        height=280,
        placeholder="Paste the full job description here...",
    )

# API key
with st.expander("🔑 OpenAI API Key (optional — for AI-powered insights)"):
    api_key = st.text_input("API Key", type="password", placeholder="sk-...",
                             value=os.getenv("OPENAI_API_KEY", ""))
    st.caption("Without API key: rule-based analysis | With API key: GPT-enhanced suggestions")

if st.button("🔍 Analyze Match", type="primary", use_container_width=True, disabled=(not job_text)):
    # Parse resume
    if resume_file:
        try:
            fname, resume_text = parse_resume(resume_file)
            st.success(f"✅ Parsed: {fname} ({len(resume_text.split())} words)")
        except Exception as e:
            st.error(f"❌ {e}")
            st.stop()
    elif resume_text_manual:
        resume_text = resume_text_manual.strip()
        st.info(f"📝 Pasted resume ({len(resume_text.split())} words)")
    else:
        st.warning("Please upload a resume or paste resume text")
        st.stop()

    if len(job_text.strip()) < 20:
        st.error("Job description is too short. Please paste the full description.")
        st.stop()

    # Analyze
    with st.spinner("🔍 Analyzing..."):
        result = analyze_match(resume_text, job_text)

    st.markdown("---")

    # ── Score Display ────────────────────────────────────
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        score = result['overall_score']
        color = '#4ade80' if score >= 70 else '#fbbf24' if score >= 50 else '#f87171'
        emoji = '🔥' if score >= 80 else '👍' if score >= 60 else '📝' if score >= 40 else '⚠️'

        st.markdown(f"""
        <div style="text-align:center;">
            <div style="position:relative;display:inline-block;">
                <svg width="180" height="180">
                    <circle cx="90" cy="90" r="80" fill="none" stroke="#1e293b" stroke-width="12"/>
                    <circle cx="90" cy="90" r="80" fill="none" stroke="{color}" stroke-width="12"
                            stroke-dasharray="502" stroke-dashoffset="{502*(1-score/100)}"
                            stroke-linecap="round" transform="rotate(-90 90 90)"/>
                </svg>
                <div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);text-align:center;">
                    <div style="font-size:2.5rem;font-weight:800;color:{color};">{score}%</div>
                    <div style="font-size:0.85rem;color:#94a3b8;">{emoji} Match</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Tabs ──────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Skill Analysis", "💡 Suggestions", "🤖 AI Insights", "📋 Details",
    ])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### ✅ Matching Skills")
            if result['matching_skills']:
                for s in result['matching_skills'][:15]:
                    st.markdown(f'<span class="skill-tag match">{s}</span>', unsafe_allow_html=True)
            else:
                st.info("No direct skill matches found. Try adding keywords from the job description.")

            st.markdown(f"**{len(result['matching_skills'])}** matching · "
                        f"**{len(result['extra_skills'])}** extra skills you have")

        with col2:
            st.markdown("### ❌ Missing Skills")
            for s in result['missing_tech'][:8]:
                st.markdown(f'<span class="skill-tag missing">{s}</span>', unsafe_allow_html=True)
            for s in result['missing_soft'][:4]:
                st.markdown(f'<span class="skill-tag missing">{s}</span>', unsafe_allow_html=True)

            if not result['missing_tech'] and not result['missing_soft']:
                st.success("No missing skills detected! Great resume alignment.")

        # Score breakdown
        st.markdown("### 📊 Score Breakdown")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Skill Match", f"{result['skill_score']}%")
        col_b.metric("Text Similarity", f"{result['tfidf_score']}%")
        col_c.metric("Overall", f"{result['overall_score']}%")

    with tab2:
        st.markdown("### 💡 Improvement Suggestions")
        for s in result['suggestions']:
            st.markdown(f"""
            <div class="suggestion-box">
                <strong>{s['icon']} {s['title']}</strong>
                <p style="color:#94a3b8;margin:0.3rem 0 0;">{s['detail']}</p>
            </div>
            """, unsafe_allow_html=True)

        # ATS keywords
        st.markdown("### 🤖 ATS Keywords")
        col_x, col_y = st.columns(2)
        with col_x:
            st.markdown("**✅ Found:**")
            for kw in result['ats_keywords_found']:
                st.markdown(f"- {kw}")
        with col_y:
            st.markdown("**❌ Missing:**")
            for kw in result['ats_keywords_missing']:
                st.markdown(f"- {kw}")

    with tab3:
        st.markdown("### 🤖 AI-Powered Analysis")
        ai_result = ai_enhance(resume_text, job_text, api_key if api_key and api_key.startswith('sk-') else None)
        if ai_result:
            st.success(ai_result)
        else:
            st.info("""
            **Enable AI insights:**
            1. Get an [OpenAI API key](https://platform.openai.com/api-keys)
            2. Enter it above and re-analyze

            **Without AI:** The rule-based analysis already provides:
            - Skill matching & gap detection
            - ATS keyword optimization
            - Actionable improvement tips
            - Match score calculation
            """)

    with tab4:
        st.markdown("### 📋 Detailed Breakdown")
        st.json({
            'overall_score': result['overall_score'],
            'skill_score': result['skill_score'],
            'tfidf_score': result['tfidf_score'],
            'matching_count': len(result['matching_skills']),
            'missing_tech_count': len(result['missing_tech']),
            'resume_word_count': result['word_count'],
            'ats_keywords_found': result['ats_keywords_found'],
            'ats_keywords_missing': result['ats_keywords_missing'],
        })

st.markdown("---")
st.markdown(
    "<center><small style='color:#64748b;'>🎯 AI Resume Analyzer · "
    "<a href='https://github.com/EkhsanFitri94'>Ekhsan Fitri</a></small></center>",
    unsafe_allow_html=True,
)
