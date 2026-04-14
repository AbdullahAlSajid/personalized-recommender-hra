import React from 'react';
import { useNavigate } from 'react-router';
import { LogOut } from 'lucide-react';
import { endSession } from '../../lib/session';

export function TopBar() {
  const navigate = useNavigate();

  const handleLogout = async () => {
    await endSession();
    navigate('/');
  };

  return (
    <div className="flex items-center justify-between px-8 py-4 bg-white/80 backdrop-blur-md sticky top-0 z-50 shadow-sm border-b border-[#e0ddd5]">
      <div className="flex items-center gap-2">
        <h1 className="text-2xl font-bold bg-gradient-to-r from-[#4ecdc4] to-[#95b8a2] bg-clip-text text-transparent">
          rec•sys
        </h1>
      </div>

      <button
        onClick={handleLogout}
        className="flex items-center gap-2 text-[#5d6875] hover:text-[#f4a261] transition-colors rounded-full px-3 py-2 hover:bg-[#faf8f5]"
        title="Avslutt økt"
      >
        <LogOut size={20} />
        <span className="text-sm font-medium">Avslutt</span>
      </button>
    </div>
  );
}
