import React from 'react';
import { useNavigate } from 'react-router';
import { endSession } from '../lib/session';
import { TopBar } from '../components/ui/TopBar';
import { Card } from '../components/ui/card';
import { Button } from '../components/ui/Button';

type Recommendation = {
  sanity_text_id?: number | string;
  title?: string;
  preview_text?: string;
  serialNumber?: number | string;
  body?: string;
};

function RecommendationSection({
  title,
  items,
  navigate,
}: {
  title: string;
  items: Recommendation[];
  navigate: ReturnType<typeof useNavigate>;
}) {
  if (!items || items.length === 0) return null;

  return (
    <section className="mb-12">
      <h2 className="text-2xl font-bold text-[#2d3142] mb-6">{title}</h2>

      <div className="grid md:grid-cols-2 gap-8">
        {items.map((book, index) => (
          <Card
            key={book.sanity_text_id || index}
            hoverEffect={true}
            onClick={() => navigate(`/reading/${book.sanity_text_id}`)}
            className="overflow-hidden flex flex-col h-full"
          >
            <div className="p-8 flex flex-col flex-1">
              <h3 className="text-2xl font-bold text-[#2d3142] mb-3">
                {book.title || `Tekst ${book.serialNumber || book.sanity_text_id}`}
              </h3>

              <p className="text-[#5d6875] mb-8 line-clamp-4 flex-1 text-lg leading-relaxed">
                {book.preview_text || 'Ingen forhåndsvisning tilgjengelig.'}
              </p>

              <Button className="w-full mt-auto">
                Jeg vil lese denne!
              </Button>
            </div>
          </Card>
        ))}
      </div>
    </section>
  );
}

export function Dashboard() {
  const navigate = useNavigate();

  const user = JSON.parse(localStorage.getItem('loggedInUser') || 'null');
  const recommendations = JSON.parse(localStorage.getItem('recommendations') || '{}');

  const difficultyRecommendations = recommendations?.difficulty || [];
  const popularityRecommendations = recommendations?.popularity || [];
  const randomRecommendations = recommendations?.random || [];

  const displayName =
    user?.username ||
    user?.name ||
    user?.full_name ||
    user?.first_name ||
    user?.student_id ||
    'Bruker';

  return (
    <div className="min-h-screen flex flex-col">
      <TopBar />

      <main className="flex-1 container mx-auto px-4 py-8 max-w-6xl">
        <div className="flex flex-col md:flex-row md:items-end justify-between gap-4 mb-12">
          <div className="space-y-2">
            <h1 className="text-3xl font-bold text-[#2d3142]">
              Hei {displayName}!
            </h1>
            <p className="text-lg text-[#5d6875]">
              Velg den teksten du har mest lyst til å lese
            </p>
          </div>

          <button
            className="text-[#5d6875] hover:text-[#4ecdc4] underline underline-offset-4 transition-colors"
            onClick={async () => {
              await endSession();
              navigate('/');
            }}
          >
            Jeg vil ikke lese nå.
          </button>
        </div>

        <RecommendationSection
          title="Anbefalt for deg basert på nivå"
          items={difficultyRecommendations}
          navigate={navigate}
        />

        <RecommendationSection
          title="Populære tekster"
          items={popularityRecommendations}
          navigate={navigate}
        />

        <RecommendationSection
          title="Tilfeldige anbefalinger"
          items={randomRecommendations}
          navigate={navigate}
        />
      </main>
    </div>
  );
}