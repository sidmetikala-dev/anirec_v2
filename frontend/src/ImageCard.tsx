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
          <p className="title">{image.title}</p>
          <p>ID: {image.anime_id}</p>
        </div>        
      </Modal>
    </>
  );
}

export default ImageCard
