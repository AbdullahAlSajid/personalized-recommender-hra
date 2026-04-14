import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router';
import { Card } from '../components/ui/card';
import { Button } from '../components/ui/Button';
import {
  InputOTP,
  InputOTPGroup,
  InputOTPSlot,
} from '../components/ui/input-otp';
import { checkSessionStatus, validatePasscode } from '../lib/session';

export function Passcode() {
  const navigate = useNavigate();
  const [code, setCode] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  // True while we check for an existing session — prevents flashing the form.
  const [checking, setChecking] = useState(true);

  useEffect(() => {
    checkSessionStatus().then((active) => {
      if (active) {
        const hasInterests = localStorage.getItem('recsys_interests');
        navigate(hasInterests ? '/dashboard' : '/interests', { replace: true });
      } else {
        setChecking(false);
      }
    });
  }, [navigate]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (code.length !== 6 || loading) return;

    setError('');
    setLoading(true);

    try {
      await validatePasscode(code);
      // Passcode valid — no DB write yet, proceed to consent
      navigate('/consent');
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Ugyldig kode. Prøv igjen.');
    } finally {
      setLoading(false);
    }
  };

  if (checking) return null;

  return (
    <div className="min-h-screen flex items-center justify-center p-4 relative z-10">
      <div className="w-full max-w-md flex flex-col items-center gap-8">

        <div className="text-center space-y-2">
          <h1 className="text-4xl font-bold text-[#2d3142]">Les i rec•sys!</h1>
          <p className="text-[#5d6875] text-lg">Skriv inn din 6-sifrede kode for å starte.</p>
        </div>

        <Card className="w-full p-8 space-y-6">
          <form onSubmit={handleSubmit} className="space-y-6 flex flex-col items-center">

            <div className="flex flex-col items-center gap-3">
              <label className="text-sm font-medium text-[#2d3142]">Tilgangskode</label>
              <InputOTP
                maxLength={6}
                value={code}
                onChange={(val) => {
                  setCode(val);
                  setError('');
                }}
                inputMode="numeric"
                pattern="[0-9]*"
              >
                <InputOTPGroup className="gap-2">
                  {[0, 1, 2, 3, 4, 5].map((i) => (
                    <InputOTPSlot
                      key={i}
                      index={i}
                      className="w-12 h-14 text-xl font-bold rounded-[12px] border-2 border-[#e0ddd5] bg-white text-[#2d3142] data-[active=true]:border-[#4ecdc4] data-[active=true]:ring-[#4ecdc4]/30"
                    />
                  ))}
                </InputOTPGroup>
              </InputOTP>
            </div>

            {error && (
              <p className="text-sm text-red-600 text-center">{error}</p>
            )}

            <Button
              type="submit"
              className="w-full"
              disabled={code.length !== 6 || loading}
            >
              {loading ? 'Sjekker kode...' : 'Gå videre'}
            </Button>

          </form>
        </Card>

      </div>
    </div>
  );
}
