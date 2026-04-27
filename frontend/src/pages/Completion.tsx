import React from 'react';
import { useNavigate } from 'react-router';
import { TopBar } from '../components/ui/TopBar';
import { Button } from '../components/ui/Button';

export function Completion() {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen flex flex-col">
      <TopBar />

      <main className="flex-1 container mx-auto px-4 flex items-center justify-center">
        <div className="max-w-xl w-full text-center space-y-8 relative z-10">

          <div className="space-y-4">
            <h1 className="text-6xl font-black tracking-tight bg-gradient-to-r from-[#4ecdc4] via-[#95b8a2] to-[#e07a5f] bg-clip-text text-transparent drop-shadow-sm">
              BRAVO
            </h1>
            <p className="text-2xl font-medium text-[#2d3142]">
              Du er ferdig med denne teksten!
            </p>
          </div>

          <div className="flex flex-col gap-4 items-center">
            <Button
              className="w-full max-w-xs text-lg py-4"
              onClick={() => navigate('/dashboard')}
            >
              Finn en ny tekst
            </Button>
          </div>

        </div>
      </main>
    </div>
  );
}
