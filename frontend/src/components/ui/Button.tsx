import React from 'react';

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'link' | 'completion' | 'outline' | 'neutral';
  children: React.ReactNode;
}

export const Button = ({ variant = 'primary', className = '', children, ...props }: ButtonProps) => {
  const baseStyles = "transition-all duration-300 rounded-[16px] px-6 py-3 font-semibold text-white flex items-center justify-center gap-2 outline-none focus:ring-2 focus:ring-[#4ecdc4]/50 disabled:opacity-50 disabled:cursor-not-allowed";
  
  const variants = {
    primary: "bg-gradient-to-r from-[#4ecdc4] to-[#95b8a2] shadow-[0_2px_8px_rgba(0,0,0,0.1)] hover:shadow-[0_12px_40px_rgba(0,0,0,0.18)] hover:-translate-y-0.5 active:translate-y-0",
    neutral: "bg-gradient-to-r from-[#8d99ae] to-[#5d6875] text-white shadow-[0_2px_8px_rgba(0,0,0,0.1)] hover:shadow-[0_12px_40px_rgba(0,0,0,0.18)] hover:-translate-y-0.5 active:translate-y-0",
    secondary: "bg-[#e0ddd5] text-[#2d3142] hover:bg-[#d1cec5] shadow-[0_2px_8px_rgba(0,0,0,0.1)]",
    outline: "bg-transparent border-2 border-[#e0ddd5] text-[#5d6875] hover:border-[#4ecdc4] hover:text-[#4ecdc4] shadow-none",
    completion: "bg-gradient-to-r from-[#f4a261] to-[#e07a5f] shadow-[0_2px_8px_rgba(0,0,0,0.1)] hover:shadow-[0_12px_40px_rgba(0,0,0,0.18)] hover:-translate-y-0.5 active:translate-y-0",
    link: "bg-transparent text-[#5d6875] shadow-none hover:text-[#4ecdc4] px-0 py-0 underline-offset-4 hover:underline",
  };

  return (
    <button 
      className={`${baseStyles} ${variants[variant]} ${className}`}
      {...props}
    >
      {children}
    </button>
  );
};
