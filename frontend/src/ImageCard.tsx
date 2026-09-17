import type { Image } from './App.tsx'

function ImageCard({ image, index }: { image: Image; index: number }) {
  const imageElement = (
    <img
      src={image.url}
      alt={image.title}
      className={`image-slot image-slot-${index + 1}`}
    />
  );

  return (
    <>
      {image.anime_id ? (
        <a 
          href={`https://myanimelist.net/anime/${image.anime_id}/`}
          target="_blank"
          rel="noopener noreferrer"
        >
          {imageElement}
        </a>
      ) : (
        imageElement
      )}
    </>
  );
}


export default ImageCard
