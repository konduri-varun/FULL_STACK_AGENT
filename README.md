# 🚀 CodeFlow AI - Professional Development Assistant

**CodeFlow AI** is a professional AI-powered development companion with 12 specialized agents for enterprise-grade full-stack solutions. Build, deploy, and scale your applications with intelligent assistance.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Next.js](https://img.shields.io/badge/Next.js-14+-black.svg)](https://nextjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)

## ✨ What is CodeFlow AI?

CodeFlow AI is your intelligent development partner that combines 12 specialized AI agents to handle every aspect of full-stack development - from backend architecture to frontend design, database optimization to deployment automation.

### 🎯 Key Highlights

- 🤖 **12 Specialized AI Agents** - Expert agents for every development need
- 💬 **Voice Assistant** - ElevenLabs integration for hands-free coding
- 🔐 **Secure Authentication** - Clerk-powered user management
- 💾 **Chat History** - MongoDB-backed session persistence
- 🎨 **Modern UI** - Beautiful Next.js interface with Tailwind CSS
- ⚡ **Real-time Responses** - FastAPI backend for instant feedback

## 🏗️ Hierarchical Multi-Agent System

## 🏗️ Hierarchical Multi-Agent System

Our intelligent coordinator delegates tasks to 12 specialized agents:

```
root_agent (Coordinator)
├── backend_agent          → FastAPI, Flask, Django, REST APIs
├── frontend_agent         → React, Vue, JavaScript integration
├── database_agent         → SQL, MongoDB, ORMs, schema design
├── deployment_agent       → Docker, Heroku, Vercel, AWS, CI/CD
├── debugging_agent        → Error analysis, troubleshooting
├── code_agent             → Code execution, testing, validation
├── search_agent           → Documentation, research, packages
├── api_testing_agent      → pytest, test automation, mocking
├── devops_agent           → Kubernetes, Terraform, monitoring
├── data_science_agent     → pandas, ML models, data pipelines
├── security_agent         → Authentication, vulnerabilities, audits
└── performance_agent      → Optimization, profiling, caching
```

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- MongoDB Atlas account (free tier)
- Google AI API key
- Clerk account for authentication

### Backend Setup

```bash
# Clone the repository
git clone https://github.com/konduri-varun/FULL_STACK_AGENT.git
cd FULL_STACK_AGENT

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
echo "GOOGLE_API_KEY=your_google_api_key" > .env
echo "MONGODB_URI=your_mongodb_uri" >> .env

# Run backend
python api.py
```

Backend will run on `http://localhost:7860`

### Frontend Setup

```bash
# Navigate to frontend
cd frontend

# Install dependencies
npm install

# Create .env.local file
echo "NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=your_clerk_key" > .env.local
echo "CLERK_SECRET_KEY=your_clerk_secret" >> .env.local
echo "NEXT_PUBLIC_API_URL=http://localhost:7860" >> .env.local

# Run frontend
npm run dev
```

Frontend will run on `http://localhost:3000`

## 📦 Tech Stack

### Backend
- **FastAPI** - High-performance Python web framework
- **Google ADK** - Agent Development Kit for AI agents
- **MongoDB** - Chat history and session storage
- **Uvicorn** - ASGI server

### Frontend
- **Next.js 14** - React framework with App Router
- **TypeScript** - Type-safe development
- **Tailwind CSS** - Utility-first styling
- **Clerk** - Authentication and user management
- **ElevenLabs** - Voice AI assistant

## 🎨 Features

### 💬 Chat Interface
- Multiple chat sessions
- Rename and delete sessions
- Session isolation (separate message history)
- Real-time streaming responses
- Persistent chat history

### 🎤 Voice Assistant
- ElevenLabs Conversational AI integration
- Hands-free interaction
- Natural language understanding
- Context-aware responses

### 🔐 Authentication
- Secure Clerk authentication
- User profile management
- Protected routes
- Session management

## 📚 Documentation

- [Quick Start Guide](QUICKSTART.md)
- [Deployment Guide](DEPLOYMENT.md)
- [API Documentation](http://localhost:7860/docs) (when backend is running)

## 🌐 Deployment

### Backend - Render
```bash
# Push to GitHub, then deploy on Render
# Render will auto-detect render.yaml
```

### Frontend - Vercel
```bash
# Deploy via Vercel CLI or GitHub integration
vercel --prod
```

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

MIT License - see LICENSE file for details

## 👨‍💻 Author

**Varun Konduri**
- GitHub: [@konduri-varun](https://github.com/konduri-varun)
- Email: kondurivarun09@gmail.com

## 🙏 Acknowledgments

- Google ADK for agent framework
- ElevenLabs for voice AI
- Clerk for authentication
- Vercel for hosting

---

**Built with ❤️ using Python, Next.js, and AI**

**Query optimization:**
```
"Optimize this SQLAlchemy query that's causing N+1 problems"
```

**Caching strategy:**
```
"Implement Redis caching for my FastAPI endpoints"
```

### Frontend Integration

**React + FastAPI connection:**
```
"Show me how to connect React to FastAPI backend with authentication"
```

**Error handling in frontend:**
```
"Handle loading states and errors in React when calling APIs"
```

### Database

**Schema design:**
```
"Design a database schema for a blog with users, posts, and comments"
```

**Query optimization:**
```
"Optimize this MongoDB query for better performance"
```

### Deployment

**Docker setup:**
```
"Create a Dockerfile for my Flask application with PostgreSQL"
```

**CI/CD pipeline:**
```
"Set up GitHub Actions CI/CD for my Django project"
```

### Debugging

**CORS errors:**
```
"I'm getting a CORS error when calling my API from React"
```

**Performance issues:**
```
"My SQLAlchemy query is slow, how can I optimize it?"
```

## 📁 Project Structure

```
FULL_STACK_AGENT/
├── app.py                      # Main Gradio application
├── requirements.txt            # Python dependencies
├── my_first_agent/
│   ├── __init__.py
│   └── agent.py               # Multi-agent system definition
├── .env                       # Environment variables (create this)
└── README.md                  # This file
```

## 🛠️ Technology Stack

- **AI Framework**: Google Agent Development Kit (ADK)
- **LLM**: Gemini 2.0 Flash
- **UI Framework**: Gradio 4.44+ with custom CSS
- **Session Management**: In-memory session service
- **Code Execution**: Built-in code executor
- **Web Search**: Google Search integration
- **Testing**: pytest, pytest-asyncio
- **Data Science**: pandas, numpy, scikit-learn
- **Security**: pyjwt, passlib, slowapi
- **HTTP Clients**: requests, httpx
- **Databases**: sqlalchemy, pymongo, redis

## 🎨 UI/UX Features

### Professional Design
- Custom gradient color scheme (Indigo/Purple)
- Inter font family for modern typography
- Smooth animations and transitions
- Responsive grid layout
- Feature cards with hover effects

### Dark/Light Mode
- Automatic theme detection
- Smooth theme transitions
- Optimized color schemes for both modes
- Custom scrollbar styling

### Interactive Elements
- Stat cards showing capabilities
- Badge components for technologies
- Example buttons with gradient hover
- Streaming response animation
- Professional chat bubbles

## 🔒 Security Notes

- Never commit your `.env` file or expose your API key
- Use environment variables for all sensitive configuration
- When deploying, use platform-specific secret management
- Validate and sanitize all user inputs in production

## 🚢 Deployment

### Heroku
```bash
# Create Procfile
echo "web: python app.py" > Procfile

# Deploy
heroku create your-app-name
heroku config:set GOOGLE_API_KEY=your-key
git push heroku main
```

### Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 7860
CMD ["python", "app.py"]
```

Build and run:
```bash
docker build -t fullstack-agent .
docker run -p 7860:7860 -e GOOGLE_API_KEY=your-key fullstack-agent
```

## 📝 Configuration

### Environment Variables
- `GOOGLE_API_KEY` (required): Your Google AI Studio API key
- `PORT` (optional): Server port (default: 7860)

### Customization

You can customize agent behavior by modifying `my_first_agent/agent.py`:
- Adjust agent instructions for different response styles
- Add new specialized agents for specific domains
- Modify delegation logic in the root agent

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 🔗 Resources

- [Google ADK Documentation](https://google.github.io/adk-docs/)
- [Gradio Documentation](https://www.gradio.app/docs/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://react.dev/)

## 📧 Support

For issues, questions, or contributions, please open an issue on GitHub.

## 🎯 Roadmap

- [x] 12 specialized AI agents
- [x] Professional UI with dark/light mode
- [x] Streaming responses
- [x] Custom CSS styling
- [x] API testing agent
- [x] DevOps agent
- [x] Data science agent
- [x] Security agent
- [x] Performance agent
- [ ] Persistent session storage with PostgreSQL
- [ ] Code snippet library and templates
- [ ] Multi-file project generator
- [ ] Real-time collaboration features
- [ ] Voice input support
- [ ] Code playground integration
- [ ] Mobile app version

---

Built with ❤️ using Google ADK, Gemini 2.0 Flash, and Gradio
