import React from 'react';
import { useNavigate } from 'react-router';
import { LogOut, User } from 'lucide-react';

export function TopBar() {
  const navigate = useNavigate();

  const user = JSON.parse(localStorage.getItem('loggedInUser') || 'null');

  const displayName =
    user?.username ||
    user?.name ||
    user?.full_name ||
    user?.first_name ||
    user?.student_id ||
    'Bruker';

  const handleLogout = () => {
    localStorage.removeItem('loggedInUser');
    localStorage.removeItem('recommendations');
    navigate('/');
  };

  return (
    <div className="flex items-center justify-between px-8 py-4 bg-white/80 backdrop-blur-md sticky top-0 z-50 shadow-sm border-b border-[#e0ddd5]">
      <div className="flex items-center gap-2">
        <h1 className="text-2xl font-bold bg-gradient-to-r from-[#4ecdc4] to-[#95b8a2] bg-clip-text text-transparent">
          rec•sys
        </h1>
      </div>

      <div className="flex items-center gap-4">
        <div className="flex items-center gap-3 bg-white px-4 py-2 rounded-full border border-[#e0ddd5] shadow-sm">
          <div className="w-8 h-8 rounded-full bg-[#e0ddd5] flex items-center justify-center text-[#5d6875]">
            <User size={16} />
          </div>
          <span className="text-sm font-medium text-[#2d3142]">
            {displayName}
          </span>
        </div>

        <button
          onClick={handleLogout}
          className="p-2 text-[#5d6875] hover:text-[#f4a261] transition-colors rounded-full hover:bg-[#faf8f5]"
          title="Logg ut"
        >
          <LogOut size={20} />
        </button>
      </div>
    </div>
  );
}