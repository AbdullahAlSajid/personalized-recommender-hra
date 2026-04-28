import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router';
import { LogOut } from 'lucide-react';
import { endSession, getSessionFeedbackTexts } from '../../lib/session';
import { Button } from './Button';
import { Card } from './card';

export function TopBar() {
  const navigate = useNavigate();
  const [endingSession, setEndingSession] = useState(false);
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [confirmSubmitting, setConfirmSubmitting] = useState(false);

  useEffect(() => {
    if (!confirmOpen) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return;
      if (confirmSubmitting) return;
      setConfirmOpen(false);
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [confirmOpen, confirmSubmitting]);

  const handleConfirmEndSession = async () => {
    if (confirmSubmitting) return;
    setConfirmSubmitting(true);
    try {
      await endSession();
      navigate('/');
    } finally {
      setConfirmSubmitting(false);
      setConfirmOpen(false);
    }
  };

  const handleLogout = async () => {
    if (endingSession) return;

    setEndingSession(true);
    try {
      const texts = await getSessionFeedbackTexts();

      if (Array.isArray(texts) && texts.length === 0) {
        setConfirmOpen(true);
        return;
      }

      navigate('/session-feedback');
    } catch {
      // If we can't check, fall back to previous behavior.
      navigate('/session-feedback');
    } finally {
      setEndingSession(false);
    }
  };

  return (
    <>
      <div className="flex items-center justify-between px-8 py-4 bg-white/80 backdrop-blur-md sticky top-0 z-50 shadow-sm border-b border-[#e0ddd5]">
        <div className="flex items-center gap-2">
          <h1 className="text-2xl font-bold bg-gradient-to-r from-[#4ecdc4] to-[#95b8a2] bg-clip-text text-transparent">
            rec•sys
          </h1>
        </div>

        <button
          onClick={handleLogout}
          disabled={endingSession}
          className="flex items-center gap-2 text-[#5d6875] hover:text-[#f4a261] transition-colors rounded-full px-3 py-2 hover:bg-[#faf8f5] disabled:opacity-50 disabled:cursor-not-allowed"
          title="Avslutt økt"
          type="button"
        >
          <LogOut size={20} />
          <span className="text-sm font-medium">Avslutt</span>
        </button>
      </div>

      {confirmOpen && (
        <div
          className="fixed inset-0 z-[100] flex items-center justify-center px-4"
          role="dialog"
          aria-modal="true"
          aria-label="Avslutt økt"
        >
          <button
            type="button"
            className="absolute inset-0 bg-black/50"
            onClick={() => {
              if (confirmSubmitting) return;
              setConfirmOpen(false);
            }}
            aria-label="Lukk"
          />

          <div
            className="relative"
            style={{ width: 400, maxWidth: 'calc(100vw - 2rem)' }}
          >
            <Card className="p-8 bg-white border border-[#e0ddd5] rounded-[24px] shadow-sm">
              <h2 className="text-xl font-semibold text-[#2d3142]">
                Avslutte økten?
              </h2>
              <p className="mt-2 text-[#5d6875]">Vil du avslutte økten nå?</p>

              <div className="mt-6 flex flex-col sm:flex-row gap-3 sm:justify-end">
                <Button
                  variant="completion"
                  onClick={() => setConfirmOpen(false)}
                  disabled={confirmSubmitting}
                  type="button"
                >
                  Fortsett å lese
                </Button>
                <Button
                  variant="primary"
                  onClick={handleConfirmEndSession}
                  disabled={confirmSubmitting}
                  type="button"
                >
                  {confirmSubmitting ? 'Avslutter...' : 'Ja, avslutt'}
                </Button>
              </div>
            </Card>
          </div>
        </div>
      )}
    </>
  );
}
