from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
from pymongo import MongoClient

import json


import requests
load_dotenv()
MONGO_URL = os.getenv("MONGO_URL")

client = MongoClient(MONGO_URL)
db = client["ai_profile_analyzer"]
users_collection = db["users"]
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
headers = {
    "Authorization": f"token {GITHUB_TOKEN}"
}
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from passlib.context import CryptContext
from jose import jwt
from datetime import datetime, timedelta

SECRET_KEY = "supersecretkey123"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


@app.get("/")
def home():
    return {"message": "Coding Profile Analyzer Backend Running"}
@app.get("/github/{username}")
def get_github_profile(username: str):
    url = f"https://api.github.com/users/{username}"
    response = requests.get(url,headers=headers)

    if response.status_code == 403:
        return {"error": "GitHub API rate limit exceeded"}

    if response.status_code == 404:
        return {"error": "GitHub user not found"}

    if response.status_code != 200:
        return {"error": "GitHub API error"}


    data = response.json()

    return {
        "username": data["login"],
        "public_repos": data["public_repos"],
        "followers": data["followers"],
        "following": data["following"],
        "account_created": data["created_at"]
    }
@app.get("/github/{username}/repos")
def get_repositories(username: str):
    url = f"https://api.github.com/users/{username}/repos"
    response = requests.get(url,headers=headers)

    if response.status_code != 200:
        return {"error": "User not found"}

    repos = response.json()

    repo_data = []

    for repo in repos:
        repo_data.append({
            "name": repo["name"],
            "stars": repo["stargazers_count"],
            "forks": repo["forks_count"],
            "language": repo["language"]
        })

    return repo_data
@app.get("/analyze/github/{username}")
def analyze_github(username: str):
    url = f"https://api.github.com/users/{username}/repos"
    response = requests.get(url,headers=headers)

    if response.status_code != 200:
        return {"error": "User not found"}

    

    repos = response.json()

    events_url = f"https://api.github.com/users/{username}/events"
    events_response = requests.get(events_url,headers=headers)

    total_commits = 0

    for repo in repos:
        repo_name = repo["name"]

    commits_url = f"https://api.github.com/repos/{username}/{repo_name}/commits?per_page=1"

    commit_response = requests.get(commits_url,headers=headers)
    if commit_response.status_code == 200:
        if "Link" in commit_response.headers:
            link_header = commit_response.headers["Link"]

            if 'rel="last"' in link_header:
                last_page = link_header.split('page=')[-1].split('>')[0]
                total_commits += int(last_page)
            else:
                total_commits += 1
        else:
            total_commits += len(commit_response.json())

    if total_commits < 20:
        consistency = "Low"
    elif total_commits < 100:
        consistency = "Moderate"
    else:
        consistency = "High"

    total_repos = len(repos)
    total_stars = sum(repo["stargazers_count"] for repo in repos)

    languages = {}
    for repo in repos:
        lang = repo["language"]
        if lang:
            languages[lang] = languages.get(lang, 0) + 1

    language_diversity = len(languages)

    score = (
    total_repos * 2 +
    total_stars * 3 +
    language_diversity * 5 +
    min(total_commits, 200) * 1
)

    # ---- Skill Classification ----
    if score < 50:
        level = "Beginner"
    elif score < 150:
        level = "Intermediate"
    else:
        level = "Advanced"
    return {
        "total_repos": total_repos,
        "total_stars": total_stars,
        "language_diversity": language_diversity,
        "github_score": score,
        "languages_used": languages,
        "skill_level": level,
        "total_commits": total_commits,
        "total_commits": total_commits,
        "consistency_level": consistency


    }
@app.get("/analyze/leetcode/{username}")
def analyze_leetcode(username: str):

    url = "https://leetcode.com/graphql"

    query = """
    query getUserProfile($username: String!) {
      matchedUser(username: $username) {
        username
        submitStats {
          acSubmissionNum {
            difficulty
            count
          }
        }
      }
    }
    """

    variables = {"username": username}

    response = requests.post(
        url,
        json={"query": query, "variables": variables}
    )

    if response.status_code != 200:
        return {"error": "Failed to fetch data"}

    data = response.json()

    user = data.get("data", {}).get("matchedUser")

    if not user:
        return {"error": "User not found"}

    stats = user["submitStats"]["acSubmissionNum"]

    easy = 0
    medium = 0
    hard = 0
    total = 0

    for item in stats:
        if item["difficulty"] == "Easy":
            easy = item["count"]
        elif item["difficulty"] == "Medium":
            medium = item["count"]
        elif item["difficulty"] == "Hard":
            hard = item["count"]
        elif item["difficulty"] == "All":
            total = item["count"]

    # Simple DSA score logic
    dsa_score = round((easy * 1) + (medium * 2) + (hard * 4))

    if dsa_score < 100:
        level = "Beginner"
    elif dsa_score < 400:
        level = "Intermediate"
    else:
        level = "Advanced"

    return {
        "total_solved": total,
        "easy": easy,
        "medium": medium,
        "hard": hard,
        "dsa_score": dsa_score,
        "skill_level": level
    }
@app.get("/analyze/profile/{github_username}/{leetcode_username}")
def analyze_full_profile(github_username: str, leetcode_username: str):

    # ---------------- GITHUB ----------------

    github_url = f"https://api.github.com/users/{github_username}/repos"
    github_response = requests.get(github_url, headers=headers)

    if github_response.status_code != 200:
        return {"error": "GitHub user not found"}

    repos = github_response.json()

    total_repos = len(repos)
    total_stars = sum(repo["stargazers_count"] for repo in repos)

    languages = {}
    total_commits = 0
    repo_details = []

    for repo in repos:
        repo_name = repo["name"]
        lang = repo["language"]

        if lang:
            languages[lang] = languages.get(lang, 0) + 1

        commits_url = f"https://api.github.com/repos/{github_username}/{repo_name}/commits?per_page=1"
        commit_response = requests.get(commits_url, headers=headers)

        repo_commit_count = 0
        if commit_response.status_code == 200:
            if "Link" in commit_response.headers:
                link_header = commit_response.headers["Link"]
                if 'rel="last"' in link_header:
                    last_page = link_header.split('page=')[-1].split('>')[0]
                    repo_commit_count = int(last_page)
                else:
                    repo_commit_count = 1
            else:
                repo_commit_count = len(commit_response.json())

        total_commits += repo_commit_count

        repo_details.append({
            "name": repo_name,
            "stars": repo["stargazers_count"],
            "forks": repo["forks_count"],
            "language": lang,
            "commit_count": repo_commit_count
        })

    language_diversity = len(languages)

    # -------- PROFESSIONAL GITHUB SCORING (0–100) --------

    repo_score = min(total_repos / 20, 1) * 25
    commit_score = min(total_commits / 300, 1) * 35
    language_score = min(language_diversity / 5, 1) * 20
    star_score = min(total_stars / 50, 1) * 20

    github_score = round(repo_score + commit_score + language_score + star_score)

    # ---------------- LEETCODE ----------------

    lc_url = "https://leetcode.com/graphql"

    query = """
    query getUserProfile($username: String!) {
      matchedUser(username: $username) {
        submitStats {
          acSubmissionNum {
            difficulty
            count
          }
        }
      }
    }
    """

    lc_response = requests.post(
        lc_url,
        json={"query": query, "variables": {"username": leetcode_username}}
    )

    if lc_response.status_code != 200:
        return {"error": "LeetCode user not found"}

    data = lc_response.json()
    user = data.get("data", {}).get("matchedUser")

    if not user:
        return {"error": "LeetCode user not found"}

    stats = user["submitStats"]["acSubmissionNum"]

    easy = medium = hard = total = 0

    for item in stats:
        if item["difficulty"] == "Easy":
            easy = item["count"]
        elif item["difficulty"] == "Medium":
            medium = item["count"]
        elif item["difficulty"] == "Hard":
            hard = item["count"]
        elif item["difficulty"] == "All":
            total = item["count"]

    # -------- DSA SCORING --------

    if total == 0:
        volume_score = difficulty_score = medium_score = 0
        dsa_score = 0
    else:
        volume_score = min(total / 300, 1) * 40
        difficulty_ratio = hard / total
        difficulty_score = min(difficulty_ratio / 0.25, 1) * 40
        medium_score = min(medium / 150, 1) * 20
        dsa_score = round(volume_score + difficulty_score + medium_score)

    overall_score = round((dsa_score * 0.6) + (github_score * 0.4))

    # -------- ORIGINAL LOGIC KEPT --------

    strengths = []
    weaknesses = []
    recommendations = []
    roadmap = []

    if hard < 20:
        weaknesses.append("Advanced DSA problem solving")
        recommendations.append("Focus on solving more Hard-level problems.")
        roadmap.append("Solve 10 Hard problems focusing on Graphs and DP.")

    if medium < 50:
        weaknesses.append("Problem solving depth")
        recommendations.append("Increase Medium-level problem count.")
        roadmap.append("Solve 30 Medium problems.")

    if total > 300:
        strengths.append("Strong DSA foundation")

    if language_diversity >= 3:
        strengths.append("Good technology exposure")
    else:
        weaknesses.append("Limited tech stack exposure")
        recommendations.append("Work with more programming languages.")
        roadmap.append("Build a project using a new technology stack.")

    if total_commits < 50:
        weaknesses.append("Low coding consistency")
        recommendations.append("Maintain regular commit activity.")
        roadmap.append("Work on 1 major project and push consistent commits.")
    else:
        strengths.append("Good coding consistency")

    if overall_score < 40:
        readiness = "Not Ready"
    elif overall_score < 60:
        readiness = "Internship Ready"
    elif overall_score < 75:
        readiness = "Placement Ready"
    elif overall_score < 90:
        readiness = "Strong Candidate"
    else:
        readiness = "Top Tier Candidate"

    # ---------------- RETURN FULL DATA ----------------

    return {
        "github_username": github_username,
        "leetcode_username": leetcode_username,

        "github_score": github_score,
        "dsa_score": dsa_score,
        "overall_score": overall_score,
        "placement_readiness": readiness,

        "github_data": {
            "total_repos": total_repos,
            "total_commits": total_commits,
            "total_stars": total_stars,
            "languages": languages,
            "repositories": repo_details
        },

        "leetcode_data": {
            "easy": easy,
            "medium": medium,
            "hard": hard,
            "total": total
        },

        "strengths": strengths,
        "weaknesses": weaknesses,
        "recommendations": recommendations,
        "roadmap": roadmap
    }






import json

@app.post("/chat")
def chat_with_ai(payload: dict):

    user_question = payload.get("question")
    profile_data = payload.get("profile")  # optional now

    if not user_question:
        return {"error": "Missing question"}

    # -------------------------
    # MODE DETECTION
    # -------------------------

    use_profile = False

    trigger_keywords = [
        "profile", "analysis", "improve", "roadmap",
        "dsa", "github", "placement", "strength",
        "weakness", "score","strongest area","focus on first","DSA skills","30-day study plan"
    ]

    for word in trigger_keywords:
        if word in user_question.lower():
            use_profile = True
            break

    # -------------------------
    # GENERAL MODE (FAST)
    # -------------------------

    if not use_profile or not profile_data:

        prompt = f"""
You are a helpful AI mentor.

Have natural conversation.
Answer clearly and concisely.
give only short and precise response.

User: {user_question}
"""

    # -------------------------
    # PROFILE MODE (SMART)
    # -------------------------

    else:

        compact_profile = {
            "github_score": profile_data.get("github_score"),
            "dsa_score": profile_data.get("dsa_score"),
            "overall_score": profile_data.get("overall_score"),
            "placement_readiness": profile_data.get("placement_readiness"),
            "strengths": profile_data.get("strengths"),
            "weaknesses": profile_data.get("weaknesses"),
            "roadmap": profile_data.get("roadmap")
        }

        profile_json = json.dumps(compact_profile, indent=2)

        prompt = f"""
You are a senior AI mentor.
give only short response.

Here is the user's coding profile:

{profile_json}

Give structured, analytical advice.


User question: {user_question}
"""

    # -------------------------
    # LLM CALL
    # -------------------------

    response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "phi3:mini",
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_predict": 500,
            "top_k": 40,
            "top_p": 0.9,
            "num_ctx": 2048
        }
    }
)


    if response.status_code != 200:
        return {"error": "Ollama not responding"}

    result = response.json()

    return {"reply": result.get("response", "No response")}

from fastapi import HTTPException

@app.post("/login")
def login(user: dict):

    email = user.get("email")
    password = user.get("password")

    db_user = users_collection.find_one({"email": email})

    if not db_user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if not pwd_context.verify(password, db_user["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    token = jwt.encode(
        {"sub": db_user["email"], "exp": expire},
        SECRET_KEY,
        algorithm=ALGORITHM
    )

    return {
        "access_token": token,
        "token_type": "bearer"
    }

@app.post("/register")
def register(user: dict):

    email = user.get("email")
    password = user.get("password")

    if not email or not password:
        return {"error": "Email and password required"}

    existing_user = users_collection.find_one({"email": email})

    if existing_user:
        return {"error": "User already exists"}

    hashed_password = pwd_context.hash(password)

    users_collection.insert_one({
        "email": email,
        "password": hashed_password,
        "created_at": datetime.utcnow()
    })

    return {"message": "User registered successfully"}
