import { useState, type FormEvent } from 'react'
import type { Image } from './App.tsx'

interface RecommendationResponse {
  recommendations: {
    anime_id: number
    title: string
    picture_link: string | null
  }[]
  error?: string
}

interface FormCardProps {
  onResultImages: (images: Image[]) => void
}

function FormCard({ onResultImages }: FormCardProps) {
  const [username, setUsername] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()

    const trimmedUsername = username.trim()
    if (!trimmedUsername) {
      onResultImages([])
      setError('Enter a MAL username.')
      return
    }

    setLoading(true)
    setError('')
    onResultImages([])

    try {
      const response = await fetch('/recs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: trimmedUsername }),
      })

      const data: RecommendationResponse = await response.json()

      if (!response.ok) {
        throw new Error(data.error || 'Could not get recommendations.')
      }

      const results = data.recommendations ?? []
      onResultImages(
        results
          .filter((recommendation) => recommendation.picture_link)
          .map((recommendation) => ({
            url: recommendation.picture_link as string,
            title: recommendation.title,
            anime_id: recommendation.anime_id,
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
        <label htmlFor="username">MAL Username</label>

        <input
          id="username"
          value={username}
          onChange={(event) => setUsername(event.target.value)}
          placeholder="Username"
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
