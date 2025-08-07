import React, { useEffect, useState } from 'react';
import axios from 'axios';
import './sessionid.css';

const SessionIdPage = () => {
  const [sessions, setSessions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [selectedSessionId, setSelectedSessionId] = useState('');
  const [chatHistory, setChatHistory] = useState([]);

  useEffect(() => {
    fetchSessions();
  }, []);

  const fetchSessions = async () => {
    setLoading(true);
    try {
      const res = await axios.get('https://legalgpt1.onrender.com/sessions');
      setSessions(res.data?.active_sessions || []);
    } catch (err) {
      alert('❌ Error fetching sessions: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const fetchChatHistory = async () => {
    if (!selectedSessionId) return alert('Please enter a session ID.');
    try {
      const res = await axios.get(
        `https://legalgpt1.onrender.com/sessions/${selectedSessionId}/history`
      );
      setChatHistory(res.data?.messages || res.data?.chat_history || []);
    } catch (err) {
      alert('❌ Error fetching chat: ' + err.message);
    }
  };

  const handleDownloadSessionIds = () => {
    const blob = new Blob([sessions.join('\n')], { type: 'text/plain' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = 'session_ids.txt';
    link.click();
  };

  const handleDownloadChat = () => {
    const formatted = chatHistory
      .map((msg) => `${msg.type === 'human' ? '👤 You' : '🤖 Bot'}: ${msg.content}`)
      .join('\n');
    const blob = new Blob([formatted], { type: 'text/plain' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `chat_${selectedSessionId}.txt`;
    link.click();
  };

  const handleDeleteSession = async () => {
    if (!selectedSessionId) return alert('Please enter a session ID to delete.');
    if (!window.confirm(`Are you sure you want to delete session ${selectedSessionId}?`)) return;

    try {
      await axios.delete(`https://legalgpt1.onrender.com/sessions/${selectedSessionId}`);
      alert(`✅ Session ${selectedSessionId} deleted.`);
      setSelectedSessionId('');
      setChatHistory([]);
      fetchSessions(); // Refresh session list
    } catch (err) {
      alert('❌ Error deleting session: ' + err.message);
    }
  };

  const handleCopySessionId = (id) => {
    navigator.clipboard.writeText(id);
    alert(`Copied: ${id}`);
  };

  return (
    <div className="home-container">
      {/* Sidebar */}
      <div className="sidebar">
        <h2>🧾 Chat History Viewer</h2>

        <input
          type="text"
          placeholder="Enter Session ID"
          value={selectedSessionId}
          onChange={(e) => setSelectedSessionId(e.target.value)}
          style={{ padding: '8px', marginBottom: '10px' }}
        />

        <button onClick={fetchChatHistory} style={{ marginBottom: '10px' }}>
          🔍 Fetch Chat
        </button>

        <button
          onClick={handleDeleteSession}
          style={{ marginBottom: '20px', backgroundColor: '#ef4444', color: 'white' }}
        >
          🗑️ Delete Session
        </button>

        {chatHistory.length > 0 && (
          <>
            <h3>💬 Chat History</h3>
            <div className="chat-history-box">
              {chatHistory.map((msg, i) => (
                <div
                  key={i}
                  className={`chat-msg ${msg.type === 'human' ? 'user' : 'bot'}`}
                >
                  <span>
                    <strong>{msg.type === 'human' ? '👤 You' : '🤖 Bot'}:</strong> {msg.content}
                  </span>
                </div>
              ))}
            </div>
            <button onClick={handleDownloadChat} style={{ marginTop: '10px' }}>
              ⬇️ Download Chat
            </button>
          </>
        )}
      </div>

      {/* Right panel */}
      <div className="chatbox">
        <h3>🆔 Active Session IDs</h3>
        <p>Total: {sessions.length}</p>
        <button
          onClick={handleDownloadSessionIds}
          className="download-btn"
          disabled={!sessions.length}
        >
          ⬇️ Download Session IDs
        </button>

        {loading ? (
          <p>Loading sessions...</p>
        ) : (
          <ul className="session-list">
            {sessions.map((sid, idx) => (
              <li key={idx} onClick={() => handleCopySessionId(sid)} title="Click to copy">
                {sid} ❐
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
};

export default SessionIdPage;
