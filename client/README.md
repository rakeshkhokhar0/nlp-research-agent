# NLP Research Agent — Frontend Client

A modern React application built with Vite and styled with Tailwind CSS for the NLP Research Agent platform.

## Technology Stack

- **React 18** — Component-based user interface library
- **Vite 6** — Fast build tool and dev server
- **Tailwind CSS 3** — Utility-first CSS framework
- **Axios** — Centralized HTTP client

## Directory Structure

```text
client/
├── public/                 # Static assets (favicons, SVG icons)
├── src/
│   ├── api/                # API client & backend conversation methods
│   │   ├── client.js       # Centralized Axios instance & health checker
│   │   ├── conversationApi.js # Backend conversation endpoints
│   │   └── conversationApi.test.js # API contract unit tests
│   ├── components/         # Modular UI components
│   │   ├── chat/           # Chat-specific views
│   │   │   ├── ChatHeader.jsx
│   │   │   ├── ChatWindow.jsx
│   │   │   ├── CitationCard.jsx
│   │   │   ├── MessageBubble.jsx
│   │   │   ├── MessageInput.jsx
│   │   │   ├── TypingIndicator.jsx
│   │   │   └── WelcomeScreen.jsx
│   │   └── layout/         # Layout scaffolding
│   │       ├── ChatLayout.jsx
│   │       └── Sidebar.jsx
│   ├── context/            # React Context providers (state management)
│   │   └── ConversationContext.jsx
│   ├── hooks/              # Custom React hooks
│   │   └── useConversation.js
│   ├── pages/              # Page-level route views
│   │   └── ChatPage.jsx
│   ├── styles/             # Tailwind directives and custom styles
│   │   └── index.css
│   ├── App.jsx             # Root application component
│   └── main.jsx            # Application entrypoint & DOM mount
├── .env.example            # Example environment configuration
├── .gitignore              # Git ignore configuration for client
├── index.html              # HTML template root
├── package.json            # Scripts and package dependencies
├── postcss.config.js       # PostCSS configuration for Tailwind
├── tailwind.config.js      # Tailwind CSS theme and content config
└── vite.config.js          # Vite plugin and dev server configuration
```

## Setup & Installation

### 1. Prerequisites
- Node.js (v18+ recommended)
- npm (v9+ recommended)

### 2. Configure Environment Variables
Copy the example environment configuration:

```bash
cp .env.example .env
```

Verify or update the backend API URL in `.env`:
```env
VITE_API_BASE_URL=http://localhost:8000/api/v1
```

### 3. Install Dependencies
```bash
npm install
```

## Development & Build Commands

### Start Development Server
Starts the Vite dev server with hot module replacement (HMR) on port `5173`:
```bash
npm run dev
```

The application will be accessible at:
```
http://localhost:5173
```

### Run Unit Tests
Executes the API client contract verification suite:
```bash
npm test
```

### Production Build
Type-checks and compiles production-ready static assets to `dist/`:
```bash
npm run build
```

### Preview Production Build
Locally preview the generated production build:
```bash
npm run preview
```

## Architecture & Integration

- **Conversation Backend Integration**:
  - `POST /api/v1/conversations`: Initializes new research conversation sessions.
  - `GET /api/v1/conversations/{id}`: Retrieves conversation session metadata.
  - `POST /api/v1/conversations/{id}/messages`: Submits user questions and receives grounded answers with evidence citations.
  - `GET /api/v1/conversations/{id}/messages`: Retrieves complete chronological message history and citations.
- **Local Conversation Tracking**:
  - Conversation metadata (`id`, `title`, timestamps) is tracked locally in `localStorage` under `nlp_research_conversations`.
  - All message content and citations are loaded on demand directly from the backend.
  - Stale or expired backend conversation sessions are handled gracefully with recovery options.
- **Strict Grounding & Citation Rendering**:
  - Assistant answers display associated `CitationCard` components with paper title, page number, quoted excerpt, relevance score, and citation order.
