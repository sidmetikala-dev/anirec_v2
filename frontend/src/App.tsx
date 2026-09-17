import { useState } from 'react'
import './App.css'
import ImageCard from './ImageCard.tsx'
import FormCard from './FormCard.tsx'

import gojo from './assets/01-gojo.jpg'
import goku from './assets/02-goku.jpg'
import naruto from './assets/03-naruto.jpg'
import luffy from './assets/04-luffy.jpg'
import tanjiro from './assets/05-tanjiro.jpg'

export interface Image {
  url: string;
  title: string;
  anime_id?: number;
}

const images: Image[] = [
  {
    url: gojo,
    title: 'Gojo',
  },
  {
    url: goku,
    title: 'Goku',
  },
  {
    url: naruto,
    title: 'Naruto',
  },
  {
    url: luffy,
    title: 'Luffy',
  },
  {
    url: tanjiro,
    title: 'Tanjiro',
  }
]

function App() {
  const [resultImages, setResultImages] = useState<Image[]>([])
  const displayedImages = resultImages.length > 0 ? resultImages : images

  return (
    <main className="gallery">
      <div className="images">
        {displayedImages.map((image, index) => (
          <ImageCard key={image.title} image={image} index={index} />
        ))}
      </div>
      <div>
        <FormCard onResultImages={setResultImages} />
      </div>
    </main>
  )
}

export default App
