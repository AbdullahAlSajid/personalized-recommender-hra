import React, { useRef, useState } from 'react';
import { useNavigate, useLocation } from 'react-router';
import { Card } from '../components/ui/card';
import { Button } from '../components/ui/Button';
import { startSession, TokenExpiredError } from '../lib/session';

export function Consent() {
  const navigate = useNavigate();
  const location = useLocation();
  const token: string = (location.state as { token?: string })?.token ?? '';
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const inFlight = useRef(false);

  // If the user lands here without a token (e.g. direct URL), send them back.
  if (!token) {
    navigate('/', { replace: true });
    return null;
  }

  const handleConsent = async () => {
    if (inFlight.current) return;
    inFlight.current = true;
    setLoading(true);
    setError('');

    try {
      await startSession(token);
      navigate('/interests');
    } catch (err) {
      if (err instanceof TokenExpiredError) {
        // Token expired (user sat on this page > 5 min) — must re-enter passcode.
        navigate('/', { replace: true, state: { expiredMessage: 'Sesjonen utløpte. Skriv inn koden på nytt.' } });
      } else {
        setError('Kunne ikke koble til serveren. Prøv igjen.');
        inFlight.current = false;
      }
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
      <div className="w-full flex flex-col items-center gap-6" style={{ maxWidth: '40rem' }}>

        <div className="text-center space-y-2">
          <h1 className="text-3xl font-bold text-[#2d3142]">Informert samtykke (skal leveres av lærer)</h1>
          <p className="text-[#5d6875]">Les nøye før samtykket registreres.</p>
        </div>

        <Card className="w-full p-8 sm:p-10 space-y-6 bg-white border-[#e0ddd5] rounded-[24px] shadow-[0_8px_30px_rgba(0,0,0,0.12)]">
          <div className="space-y-6 text-[#2d3142] text-sm leading-relaxed">
            <p className="font-semibold">Samtykket registreres av lærer og gjelder studentens deltakelse.</p>

            <div className="space-y-2">
              <p className="font-semibold">Kort om prosjektet</p>
              <p className="text-[#5d6875]">
                Dette er en del av et forskningsprosjekt ved universitetet. Målet er å forbedre et
                personalisert leseanbefalingssystem.
              </p>
            </div>

            <div className="space-y-2">
              <p className="font-semibold">Hva registreres</p>
              <ul className="list-disc pl-5 text-[#5d6875] space-y-1">
                <li>Hvilke tekster studenten leser</li>
                <li>Studentens svar på spørsmål</li>
              </ul>
            </div>

            <div className="space-y-2">
              <p className="font-semibold">Frivillighet og personvern</p>
              <ul className="list-disc pl-5 text-[#5d6875] space-y-1">
                <li>Deltakelsen er frivillig</li>
                <li>Dataene lagres uten navn eller andre identifiserende opplysninger</li>
                <li>Studenten kan avslutte når som helst</li>
              </ul>
            </div>

            <p className="font-bold border-l-4 border-[#e0ddd5] pl-6">
              Hvis foresatte har gitt samtykke til at studenten kan delta, velg «Samtykke gitt».
              Hvis ikke, velg «Samtykke ikke gitt».
            </p>
          </div>

          {error && (
            <p className="text-sm text-red-600 text-center">{error}</p>
          )}

          <div className="grid grid-cols-2 gap-3 pt-0">
            <Button
              onClick={handleConsent}
              disabled={loading}
              variant="primary"
              className="w-full"
            >
              {loading ? 'Starter økt...' : 'Samtykke gitt'}
            </Button>

            <Button
              onClick={handleDecline}
              disabled={loading}
              variant="completion"
              className="w-full"
            >
              Samtykke ikke gitt
            </Button>
          </div>
        </Card>

      </div>
    </div>
  );
}
