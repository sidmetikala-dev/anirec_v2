import { useState, type FormEvent } from 'react'
import type { Image } from './App.tsx'

interface RecommendationResponse {
  recommendations: {
    anime_id: number
    title: string
    picture_link: string | null
    explanation: string
    synopsis: string
    genres: string[]
    mean_rating: number
    num_episodes: number
    num_list_users: number
  }[]
  error?: string
}

interface FormCardProps {
  onResultImages: (images: Image[]) => void
}

function FormCard({ onResultImages }: FormCardProps) {
  const [prompt, setPrompt] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()

    const trimmedPrompt = prompt.trim()
    if (!trimmedPrompt) {
      onResultImages([])
      setError('Enter a recommendation request or MAL username.')
      return
    }

    setLoading(true)
    setError('')
    onResultImages([])

    try {
      const params = new URLSearchParams({ prompt: trimmedPrompt })
      const response = await fetch(`/agent?${params}`)

      const data: RecommendationResponse = await response.json()

      if (!response.ok || data.error) {
        throw new Error(data.error || 'Could not get recommendations.')
      }

      const results = data.recommendations ?? []
      if (results.length === 0) {
        throw new Error('No matching recommendations found. Try a different request.')
      }
      onResultImages(
        results
          .filter((recommendation) => recommendation.picture_link)
          .slice(0, 5)
          .map((recommendation) => ({
            url: recommendation.picture_link as string,
            title: recommendation.title,
            anime_id: recommendation.anime_id,
            agent_explanation: recommendation.explanation,
            anime_synopsis: recommendation.synopsis,
            anime_genres: recommendation.genres,
            anime_mean_rating: recommendation.mean_rating,
            anime_num_episodes: recommendation.num_episodes,
            members: recommendation.num_list_users,
          }))
      )
    } catch (error) {
      setError(
        error instanceof Error ? error.message : 'Something went wrong.'
      )
    } finally {
      setLoading(false)
    }
  }

  return (
    <>
      <h1>AniRecs</h1>

      <form className="form-card" onSubmit={handleSubmit}>
        <label htmlFor="prompt">What would you like to watch?</label>

        <input
          id="prompt"
          value={prompt}
          onChange={(event) => setPrompt(event.target.value)}
          placeholder="Anime like Naruto, or your MAL username"
          maxLength={4000}
          disabled={loading}
        />

        <button className="fetch" disabled={loading}>
          <b>{loading ? 'Loading...' : 'Get Recs'}</b>
        </button>
        {error && <p className="status-message">{error}</p>}
      </form>
    </>
  )
}

export default FormCard
