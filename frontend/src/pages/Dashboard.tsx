import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router';
import { RefreshCw } from 'lucide-react';
import {
  getSessionRecommendations,
  refreshSessionRecommendations,
  logSessionEvent,
  SessionExpiredError,
  BACKEND_URL,
  type SessionRecommendation,
} from '../lib/session';
import { TopBar } from '../components/ui/TopBar';
import { Card } from '../components/ui/card';
import { Button } from '../components/ui/Button';

type Recommendation = {
  text_id?: string;
  sanity_text_id?: number | string;
  title?: string;
  broad_topics?: string[];
  first_image_url?: string | null;
  preview_text?: string | null;
  serialNumber?: number | string;
  body?: string | null;
};

function normalizePreviewText(value: string): string {
  return value.replace(/\s+/g, ' ').trim();
}

function derivePreviewText(book: Recommendation): string {
  const explicit = typeof book.preview_text === 'string' ? book.preview_text.trim() : '';
  if (explicit) return explicit;

  let body = typeof book.body === 'string' ? book.body : '';
  body = body.replace(/\r\n/g, '\n').trim();

  const title = typeof book.title === 'string' ? book.title.trim() : '';
  if (title && body) {
    const lines = body
      .split('\n')
      .map((l) => l.trim())
      .filter((l) => l.length > 0);

    if (lines.length > 0 && lines[0].toLowerCase() === title.toLowerCase()) {
      lines.shift();
      body = lines.join(' ');
    } else if (body.toLowerCase().startsWith(title.toLowerCase())) {
      body = body.slice(title.length);
    }
  }

  const normalized = normalizePreviewText(body);
  if (!normalized) return 'Ingen forhåndsvisning tilgjengelig.';

  const maxChars = 250;
  if (normalized.length > maxChars) {
    return `${normalized.slice(0, maxChars).trimEnd()}....`;
  }
  return normalized;
}

function resolveImageUrl(firstImageUrl?: string | null): string | null {
  if (!firstImageUrl) return null;
  if (firstImageUrl.startsWith('http://') || firstImageUrl.startsWith('https://')) {
    return firstImageUrl;
  }
  if (firstImageUrl.startsWith('/')) {
    return `${BACKEND_URL}${firstImageUrl}`;
  }
  return `${BACKEND_URL}/${firstImageUrl}`;
}

function resolveThumbnailUrl(url?: string | null): string | null {
  if (!url) return null;

  try {
    const resolved = new URL(url, BACKEND_URL);
    if (
      !resolved.pathname.startsWith('/images/') ||
      resolved.pathname.startsWith('/images/thumbs/')
    ) {
      return null;
    }

    resolved.pathname = resolved.pathname.replace('/images/', '/images/thumbs/');
    return resolved.toString();
  } catch {
    return null;
  }
}

function DashboardImage({
  src,
  alt,
  prioritize = false,
  ...props
}: React.ImgHTMLAttributes<HTMLImageElement> & {
  prioritize?: boolean;
}) {
  const fallbackSrc = src ?? null;
  const thumbnailSrc = resolveThumbnailUrl(fallbackSrc);
  const preferredSrc = thumbnailSrc ?? fallbackSrc;
  const [currentSrc, setCurrentSrc] = useState(preferredSrc ?? '');

  useEffect(() => {
    setCurrentSrc(preferredSrc ?? '');
  }, [preferredSrc]);

  if (!fallbackSrc) return null;

  return (
    <img
      {...props}
      src={currentSrc}
      alt={alt}
      loading={prioritize ? 'eager' : 'lazy'}
      fetchPriority={prioritize ? 'high' : 'auto'}
      decoding="async"
      onError={() => {
        if (currentSrc !== fallbackSrc) {
          setCurrentSrc(fallbackSrc);
        }
      }}
    />
  );
}

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
      <h2 className="text-2xl font-bold text-[#2d3142] mb-4">{title}</h2>

      <div className="grid md:grid-cols-2 gap-8">
        {items.map((book, index) => (
          (() => {
            const resolvedImageUrl = resolveImageUrl(book.first_image_url);
            const shouldPrioritize = index < 2;

            return (
          <Card
            key={book.text_id || book.sanity_text_id || index}
            className="overflow-hidden flex flex-col h-full border-2 shadow-md"
          >
            <div
              className="w-full overflow-hidden bg-[#f3f1eb] shrink-0 border-b"
              style={{ aspectRatio: '4 / 3' }}
            >
              {resolvedImageUrl && (
                <DashboardImage
                  src={resolvedImageUrl}
                  alt={book.title || 'Tekst bilde'}
                  className="w-full h-full"
                  prioritize={shouldPrioritize}
                  style={{ objectFit: 'cover', objectPosition: 'center' }}
                />
              )}
            </div>
            <div className="px-8 pt-5 pb-6 flex flex-col flex-1">
              <h3 className="text-2xl font-bold text-[#2d3142] mb-3">
                {book.title || `Tekst ${book.serialNumber || book.text_id || book.sanity_text_id}`}
              </h3>

              {Array.isArray(book.broad_topics) && book.broad_topics.length > 0 && (
                <div className="flex flex-wrap gap-2 mb-4">
                  {book.broad_topics.map((topic) => (
                    <span
                      key={topic}
                      className="text-xs font-semibold px-3 py-1 rounded-full bg-[#e9f8f7] text-[#2f7f79]"
                    >
                      {topic}
                    </span>
                  ))}
                </div>
              )}

              <p className="text-[#5d6875] mb-6 line-clamp-4 flex-1 text-lg leading-relaxed">
                {derivePreviewText(book)}
              </p>

              <Button
                className="w-full mt-auto mb-3"
                onClick={() => {
                  const selectedId = book.text_id;
                  if (typeof selectedId === 'string' && selectedId.length > 0) {
                    void logSessionEvent({
                      event_type: 'dashboard_text_selected',
                      text_id: selectedId,
                      metadata: {
                        card_index: index,
                        shown_text_ids: items
                          .map((it) => it.text_id)
                          .filter((id): id is string => typeof id === 'string' && id.length > 0)
                          .slice(0, 2),
                      },
                    }).catch(() => {
                      // best-effort analytics
                    });
                  }

                  navigate(`/reading/${book.text_id || book.sanity_text_id}`);
                }}
              >
                Jeg vil lese denne!
              </Button>
            </div>
          </Card>
            );
          })()
        ))}
      </div>
    </section>
  );
}

export function Dashboard() {
  const navigate = useNavigate();
  const [recommendations, setRecommendations] = useState<SessionRecommendation[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const user = JSON.parse(localStorage.getItem('loggedInUser') || 'null');

  useEffect(() => {
    let active = true;

    const loadRecommendations = async () => {
      setLoading(true);
      setError(null);
      try {
        const texts = await getSessionRecommendations();
        if (!active) return;
        setRecommendations(texts);
      } catch (err) {
        if (!active) return;
        if (err instanceof SessionExpiredError) { navigate('/'); return; }
        setError(err instanceof Error ? err.message : 'Kunne ikke hente anbefalinger.');
      } finally {
        if (active) {
          setLoading(false);
        }
      }
    };

    void loadRecommendations();
    return () => {
      active = false;
    };
  }, []);

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
        <div className="flex flex-col md:flex-row md:items-end justify-between gap-4 mb-8">
          <div className="space-y-2">
            <h1 className="text-3xl font-bold text-[#2d3142]">
              Hei {displayName}!
            </h1>
            <p className="text-lg text-[#5d6875]">
              Velg den teksten du har mest lyst til å lese
            </p>
          </div>

          <Button
            variant="completion"
            disabled={loading || refreshing || recommendations.length === 0}
            onClick={async () => {
              const shownTextIds = recommendations
                .map((r) => r.text_id)
                .filter((id): id is string => typeof id === 'string' && id.length > 0);
              const payloadIds = shownTextIds.slice(0, 2);

              if (payloadIds.length === 0) return;

              setRefreshing(true);
              setError(null);
              try {
                const next = await refreshSessionRecommendations(payloadIds);
                setRecommendations(next);
              } catch (err) {
                if (err instanceof SessionExpiredError) { navigate('/'); return; }
                setError(err instanceof Error ? err.message : 'Kunne ikke oppdatere anbefalingene.');
              } finally {
                setRefreshing(false);
              }
            }}
          >
            <RefreshCw size={18} className={refreshing ? 'animate-spin' : ''} />
            {refreshing ? 'Oppdaterer...' : 'Oppdater anbefalinger'}
          </Button>
        </div>

        {loading && (
          <p className="text-[#5d6875]">Laster anbefalinger...</p>
        )}

        {!loading && error && (
          <p className="text-red-600">{error}</p>
        )}

        {!loading && !error && recommendations.length === 0 && (
          <p className="text-[#5d6875]">Ingen anbefalinger tilgjengelig akkurat nå.</p>
        )}

        {!loading && !error && recommendations.length > 0 && (
          <RecommendationSection
            title="Anbefalt for deg"
            items={recommendations}
            navigate={navigate}
          />
        )}
      </main>
    </div>
  );
}