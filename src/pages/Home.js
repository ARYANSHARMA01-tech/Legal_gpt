import React, { useState } from 'react';
import axios from 'axios';
import ScrollToBottom from 'react-scroll-to-bottom';
import ReactMarkdown from 'react-markdown';
import './home.css';

const Home = () => {
  const [file, setFile] = useState(null);
  const [userQuery, setUserQuery] = useState('');
  const [analysis, setAnalysis] = useState(null);
  const [chat, setChat] = useState([]);
  const [userMessage, setUserMessage] = useState('');
  const [loading, setLoading] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const [uploaded, setUploaded] = useState(false);

  const resetSession = () => {
    setSessionId(null);
    setUploaded(false);
    setChat([]);
    setAnalysis(null);
  };

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    resetSession(); // resets everything on new file
  };

  const handleUpload = async () => {
    if (!file || !userQuery) {
      alert('Please upload a document and enter a query.');
      return;
    }

    const summaryForm = new FormData();
    summaryForm.append('file', file);
    summaryForm.append('user_query', userQuery);

    const ragForm = new FormData();
    ragForm.append('files', file);

    try {
      setLoading(true);

      // Summary API (optional)
      const summaryRes = await axios.post(
        'https://legalgpt-duvt.onrender.com/process',
        summaryForm,
        { headers: { 'Content-Type': 'multipart/form-data' } }
      );
      setAnalysis(summaryRes.data);

      // Upload document to chatbot backend
      const ragRes = await axios.post(
        'https://legalgpt1.onrender.com/upload',
        ragForm,
        { headers: { 'Content-Type': 'multipart/form-data' } }
      );

      const sid = ragRes.data?.session_id;
      if (!sid) throw new Error('No session_id returned from backend.');
      setSessionId(sid);
      setUploaded(true);

      setChat([
        { sender: 'bot', message: '📄 Document uploaded. Ask your legal questions below.' },
        { sender: 'bot', message: `🆔 Session ID: ${sid}` },
        { sender: 'bot', message: `📁 File Uploaded: ${file.name}` },
      ]);
    } catch (err) {
      const errMsg = err.response?.data?.detail || err.message || 'Upload error';
      alert(`❌ Upload failed: ${errMsg}`);
    } finally {
      setLoading(false);
    }
  };

  const handleChatSend = async () => {
    if (!userMessage.trim()) return;

    const newChat = [...chat, { sender: 'user', message: userMessage }];
    setChat(newChat);
    setUserMessage('');

    if (!sessionId) {
      setChat([
        ...newChat,
        { sender: 'bot', message: '⚠️ No session found. Please upload a document first.' },
      ]);
      return;
    }

    try {
      const payload = {
        session_id: sessionId,
        question: userMessage,
        groq_api_key: process.env.REACT_APP_GROQ_API_KEY,
      };

      const res = await axios.post(
        'https://legalgpt1.onrender.com/chat',
        payload,
        { headers: { 'Content-Type': 'application/json' } }
      );

      const botAnswer = res.data?.answer || '⚠️ No response from bot.';
      setChat([...newChat, { sender: 'bot', message: botAnswer }]);
    } catch (err) {
      const errText = err.response?.data?.detail || err.message || 'Chat error';
      setChat([...newChat, { sender: 'bot', message: `❌ Chat error: ${errText}` }]);
    }
  };

  const MarkdownRenderer = ({ content }) => (
    <div className="markdown-box">
      <ReactMarkdown>{content}</ReactMarkdown>
    </div>
  );

  return (
    <div className="home-container">
      {/* Sidebar */}
      <div className="sidebar">
        <h2>📄 LegalGPT</h2>
        <input type="file" onChange={handleFileChange} />
        <input
          type="text"
          value={userQuery}
          onChange={(e) => setUserQuery(e.target.value)}
          placeholder="Enter your legal query (e.g. Summary)"
          style={{ marginTop: '8px', padding: '6px', width: '100%' }}
        />
        <button onClick={handleUpload} disabled={loading}>
          {loading ? 'Analyzing...' : 'Analyze Document'}
        </button>

        {analysis && (
          <div className="analysis-box">
            <h3>📌 Result</h3>
            {analysis.result ? (
              <MarkdownRenderer content={analysis.result} />
            ) : (
              <p>❌ No summary available</p>
            )}
            <h3>💡 Full Response (Debug)</h3>
            <pre style={{ fontSize: '12px' }}>{JSON.stringify(analysis, null, 2)}</pre>
          </div>
        )}
      </div>

      {/* Chatbox */}
      <div className="chatbox">
        <ScrollToBottom className="chat-window">
          {chat.length === 0 ? (
            <div className="chat-placeholder">
              <h2>👋 Welcome to LegalGPT</h2>
              <p>Upload a document on the left to get started.</p>
              <p>Then ask me any legal questions here.</p>
            </div>
          ) : (
            chat.map((entry, index) => (
              <div key={index} className={`chat-msg ${entry.sender}`}>
                <span>{entry.message}</span>
              </div>
            ))
          )}
        </ScrollToBottom>

        <div className="chat-input">
          <input
            type="text"
            value={userMessage}
            onChange={(e) => setUserMessage(e.target.value)}
            placeholder="Ask LegalGPT anything..."
          />
          <button onClick={handleChatSend} disabled={loading}>
            Send
          </button>
        </div>
      </div>
    </div>
  );
};

export default Home;
