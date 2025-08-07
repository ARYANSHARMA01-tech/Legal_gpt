import React, { useState } from 'react';
import './blog.css';

const BlogPage = () => {
  const blogPosts = [
    {
      id: 1,
      title: 'LegalGPT: Transforming Legal Docs with AI',
      date: 'July 25, 2025',
      content: `LegalGPT simplifies document understanding with AI-based summarization and Q&A features.
It allows legal professionals to extract key clauses, summaries, and even chat with the document in natural language.

Our Groq-powered backend handles queries at lightning speed using RAG (Retrieval Augmented Generation) pipelines.

Stay tuned as we add more tools like contract clause detection and section tagging.`
    },
    {
      id: 2,
      title: 'How Session IDs Help Track AI Chats',
      date: 'July 24, 2025',
      content: `Session IDs are essential for maintaining conversation continuity in chatbots.
They help users resume chats, download histories, and provide context-aware answers from uploaded files.

LegalGPT uses UUID-based session IDs for security and tracking without storing any personal data.`
    },
    {
      id: 3,
      title: 'Behind the Scenes: How RAG & Groq Power LegalGPT',
      date: 'July 23, 2025',
      content: `RAG (Retrieval Augmented Generation) is a technique that allows LLMs to query real documents before answering.

By combining RAG with Groq’s ultra-fast inference API, LegalGPT answers complex legal questions with relevant PDF context.

This is the foundation for LegalGPT’s accuracy in contract reviews and offer letter evaluations.`
    },
  ];

  const [expandedPostId, setExpandedPostId] = useState(null);

  const toggleExpand = (id) => {
    setExpandedPostId(prevId => (prevId === id ? null : id));
  };

  return (
    <div className="blog-page">
      <header className="blog-header">
        <h1>📝 LegalGPT Blog</h1>
        <p>Latest news, updates & insights from our team</p>
      </header>

      {/* 👥 Contributors Section */}
      <div className="contributors-box">
        <div className="contributor-card">
          <h3>Parth Sharma</h3>
        </div>
        <div className="contributor-card">
          <h3>Aryan Sharma</h3>
        </div>
      </div>

      {/* 📝 Blog Posts */}
      <div className="blog-posts">
        {blogPosts.map((post) => (
          <div className="blog-card" key={post.id}>
            <h2>{post.title}</h2>
            <p className="date">{post.date}</p>
            <p>
              {expandedPostId === post.id
                ? post.content
                : post.content.substring(0, 120) + '...'}
            </p>
            <button onClick={() => toggleExpand(post.id)}>
              {expandedPostId === post.id ? 'Read Less' : 'Read More'}
            </button>
          </div>
        ))}
      </div>
    </div>
  );
};

export default BlogPage;
