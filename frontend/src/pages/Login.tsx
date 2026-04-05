import React, { useState } from 'react';
import { useNavigate } from 'react-router';
import { Card } from '../components/ui/Card';
import { Input } from '../components/ui/Input';
import { Button } from '../components/ui/Button';

export function Login() {
  const navigate = useNavigate();
  const [studentId, setStudentId] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      const response = await fetch('http://127.0.0.1:5000/api/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ student_id: studentId }),
      });

      const data = await response.json();

      if (!response.ok || !data.success) {
        setError(data.message || 'Innlogging mislyktes');
        return;
      }

      localStorage.setItem('loggedInUser', JSON.stringify(data.user));
      localStorage.setItem('recommendations', JSON.stringify(data.recommendations || {}));

      navigate('/interests');
    } catch (err) {
      setError('Kunne ikke koble til backend');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center p-4 relative z-10">
      <div className="w-full max-w-md flex flex-col items-center gap-8">
        <div className="text-center space-y-2">
          <h1 className="text-4xl font-bold text-[#2d3142]">Les i rec•sys!</h1>
          <p className="text-[#5d6875] text-lg">Din personlige lesevenn.</p>
        </div>

        <Card className="w-full p-8 space-y-6">
          <form onSubmit={handleLogin} className="space-y-6">
            <div className="space-y-4">
              <Input
                label="Student ID"
                placeholder="Skriv inn student ID"
                value={studentId}
                onChange={(e) => setStudentId(e.target.value)}
              />
            </div>

            {error && <p className="text-sm text-red-600">{error}</p>}

            <Button type="submit" className="w-full" disabled={loading}>
              {loading ? 'Logger inn...' : 'Logg inn'}
            </Button>
          </form>
        </Card>
      </div>
    </div>
  );
}