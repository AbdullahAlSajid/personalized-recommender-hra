import React, { useRef, useState } from 'react';
import { useNavigate } from 'react-router';
import { Card } from '../components/ui/card';
import { Button } from '../components/ui/Button';
import { startSession, getSessionId } from '../lib/session';

export function Consent() {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  // useRef guard prevents a second /start call if the user double-clicks
  // or navigates back and clicks again before the first resolves.
  const inFlight = useRef(false);

  const handleConsent = async () => {
    // Session already started (e.g. user navigated back) — skip /start.
    if (getSessionId()) {
      navigate('/interests');
      return;
    }
    if (inFlight.current) return;
    inFlight.current = true;
    setLoading(true);
    setError('');

    try {
      await startSession();
      navigate('/interests');
    } catch {
      setError('Kunne ikke koble til serveren. Prøv igjen.');
      inFlight.current = false;
    } finally {
      setLoading(false);
    }
  };

  const handleDecline = () => {
    // No DB row was created — just go back. Nothing to clean up.
    navigate('/');
  };

  return (
    <div className="min-h-screen flex items-center justify-center p-4 relative z-10">
      <div className="w-full max-w-lg flex flex-col items-center gap-8">

        <div className="text-center space-y-2">
          <h1 className="text-3xl font-bold text-[#2d3142]">Informert samtykke</h1>
          <p className="text-[#5d6875]">Les nøye før du fortsetter.</p>
        </div>

        <Card className="w-full p-8 space-y-6">
          <div className="space-y-4 text-[#2d3142] text-sm leading-relaxed">
            <p>
              Dette er en del av et forskningsprosjekt ved universitetet. Vi registrerer
              hvilke tekster du leser og svarene dine på spørsmål, for å forbedre et
              personalisert leseanbefalingssystem.
            </p>
            <p>
              Deltakelsen er frivillig. Dataene lagres uten navn eller andre identifiserende
              opplysninger. Du kan avslutte når som helst.
            </p>
            <p className="font-medium">
              Ved å klikke «Jeg samtykker» bekrefter du at du har forstått dette og
              ønsker å delta.
            </p>
          </div>

          {error && (
            <p className="text-sm text-red-600 text-center">{error}</p>
          )}

          <div className="flex flex-col gap-3 pt-2">
            <Button
              onClick={handleConsent}
              disabled={loading}
              className="w-full"
            >
              {loading ? 'Starter økt...' : 'Jeg samtykker'}
            </Button>

            <button
              onClick={handleDecline}
              disabled={loading}
              className="text-[#5d6875] hover:text-[#f4a261] underline underline-offset-4 transition-colors text-sm"
            >
              Nei takk, jeg vil ikke delta
            </button>
          </div>
        </Card>

      </div>
    </div>
  );
}
