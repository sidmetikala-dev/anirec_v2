import type { Image } from './App.tsx'
import Modal from './Modal.tsx'
import { useCallback, useState } from "react";

function ImageCard({ image, index }: { image: Image; index: number }) {
  const [modal, setModal] = useState(false);
  const imageElement = (
    <img
      src={image.url}
      alt={image.title}
      className={`image-slot image-slot-${index + 1}`}
    />
  );

  const toggleModal = useCallback(() => {
    setModal((isOpen) => !isOpen)
  }, [])

  return (
    <>
      {image.anime_id ? (
        <button
          type="button"
          className="image-card-button"
          aria-label={`View details for ${image.title}`}
          onClick={toggleModal} 
        >
          {imageElement}
        </button>
      ) : (
        imageElement
      )}

      <Modal isOpen={modal} onClose={toggleModal}>
        <div className="modal-poster-wrapper">
          <a href={`https://myanimelist.net/anime/${image.anime_id}/`}
           target="_blank"
           rel="noopener noreferrer">
            <img className="modal-poster" src={image.url} alt={image.title} />
          </a>
        </div>
        <div className="modal-info">
          <a className="modal-anime-title" href={`https://myanimelist.net/anime/${image.anime_id}/`} 
           data-letters={image.title} 
           target="_blank" 
           rel="noopener noreferrer">
            {image.title}
          </a>
          <div className="modal-anime-details" aria-label="Anime statistics">
            <span className="detail-pill detail-episodes">{image.anime_num_episodes ? `${image.anime_num_episodes} episodes` : 'Episodes unavailable'}</span>
            <span className="detail-pill detail-rating">★ {image.anime_mean_rating?.toFixed(2) ?? '—'} <span>MAL rating</span></span>
            <span className="detail-pill detail-members">♡ {image.members?.toLocaleString('en-US') ?? '—'} <span>members</span></span>
          </div>
          <section className="modal-section">
            <h2>Genres</h2>
            <div className="modal-anime-genres">
              {image.anime_genres?.length
                ? image.anime_genres.map((genre) => <span className="genre-pill" key={genre}>{genre}</span>)
                : <p>Not available.</p>}
            </div>
          </section>
          <div className="modal-text">
            <section className="modal-section">
              <h2>Synopsis</h2>
              <p>{image.anime_synopsis || 'Not available.'}</p>
            </section>
            <section className="modal-section modal-reason">
              <h2>Why this anime?</h2>
              <p>{image.agent_explanation || 'Not available.'}</p>
            </section>
          </div>
          
        </div>        
      </Modal>
    </>
  );
}

export default ImageCard
