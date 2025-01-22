import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './IntroPage.css';

const IntroPage = () => {
  const [showSubtitle, setShowSubtitle] = useState(false);
  const [showClickMessage, setShowClickMessage] = useState(false);

  useEffect(() => {
    const subtitleDelay = 2000; // Delay before showing subtitle
    const clickMessageDelay = 3000; // Delay before showing click message

    setTimeout(() => {
      setShowSubtitle(true);
    }, subtitleDelay);

    setTimeout(() => {
      setShowClickMessage(true);
    }, clickMessageDelay);
  }, []);
  const navigate = useNavigate();

  const handleProceed = () => {
    navigate('/form');
  };

  return (
    <div className="intro-page" onClick={handleProceed}>
      <div className="intro-text">
        <h1 className='typing-effect'>Breathe Better</h1>
        {showSubtitle && <p className='subtitle'>Welcome to your personalized air quality prediction app!</p>}
      </div>
      {showClickMessage && <p className="click-to-continue">Click anywhere to begin</p>}
    </div>
  );
};

export default IntroPage;
