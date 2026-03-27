"use client"

import { useState, useRef, useCallback } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Upload, Sparkles, ImageIcon, Zap, RotateCcw } from "lucide-react"

const FASHION_LABELS = [
  "T-shirt/Top",
  "Trouser",
  "Pullover",
  "Dress",
  "Coat",
  "Sandal",
  "Shirt",
  "Sneaker",
  "Bag",
  "Ankle Boot",
]

const LABEL_ICONS: Record<string, string> = {
  "T-shirt/Top": "👕",
  "Trouser": "👖",
  "Pullover": "🧥",
  "Dress": "👗",
  "Coat": "🧥",
  "Sandal": "🩴",
  "Shirt": "👔",
  "Sneaker": "👟",
  "Bag": "👜",
  "Ankle Boot": "🥾",
}

export function FashionClassifier() {
  const [originalImage, setOriginalImage] = useState<string | null>(null)
  const [grayscaleImage, setGrayscaleImage] = useState<string | null>(null)
  const [prediction, setPrediction] = useState<string | null>(null)
  const [confidence, setConfidence] = useState<number | null>(null)
  const [isClassifying, setIsClassifying] = useState(false)
  const [isDragging, setIsDragging] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)

  const convertToGrayscale = useCallback((imageSrc: string): Promise<string> => {
    return new Promise((resolve) => {
      const img = new Image()
      img.crossOrigin = "anonymous"
      img.onload = () => {
        const canvas = canvasRef.current
        if (!canvas) return
        
        const ctx = canvas.getContext("2d")
        if (!ctx) return
        
        // Set canvas to 28x28 (Fashion MNIST size)
        canvas.width = 28
        canvas.height = 28
        
        // Draw and resize image
        ctx.drawImage(img, 0, 0, 28, 28)
        
        // Get image data and convert to grayscale
        const imageData = ctx.getImageData(0, 0, 28, 28)
        const data = imageData.data
        
        for (let i = 0; i < data.length; i += 4) {
          const gray = data[i] * 0.299 + data[i + 1] * 0.587 + data[i + 2] * 0.114
          data[i] = gray
          data[i + 1] = gray
          data[i + 2] = gray
        }
        
        ctx.putImageData(imageData, 0, 0)
        resolve(canvas.toDataURL())
      }
      img.src = imageSrc
    })
  }, [])

  const handleImageUpload = useCallback(async (file: File) => {
    const reader = new FileReader()
    reader.onload = async (e) => {
      const imageSrc = e.target?.result as string
      setOriginalImage(imageSrc)
      setPrediction(null)
      setConfidence(null)
      
      const grayscale = await convertToGrayscale(imageSrc)
      setGrayscaleImage(grayscale)
    }
    reader.readAsDataURL(file)
  }, [convertToGrayscale])

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      handleImageUpload(file)
    }
  }

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    const file = e.dataTransfer.files[0]
    if (file && file.type.startsWith("image/")) {
      handleImageUpload(file)
    }
  }, [handleImageUpload])

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = () => {
    setIsDragging(false)
  }

  const classifyImage = async () => {
    if (!grayscaleImage) return
    
    setIsClassifying(true)
    
    // Simulate ML model prediction with realistic delay
    await new Promise((resolve) => setTimeout(resolve, 1500))
    
    // Mock prediction (in real app, this would call an API)
    const randomIndex = Math.floor(Math.random() * FASHION_LABELS.length)
    const mockConfidence = 75 + Math.random() * 24 // 75-99%
    
    setPrediction(FASHION_LABELS[randomIndex])
    setConfidence(mockConfidence)
    setIsClassifying(false)
  }

  const resetClassifier = () => {
    setOriginalImage(null)
    setGrayscaleImage(null)
    setPrediction(null)
    setConfidence(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ""
    }
  }

  return (
    <div className="min-h-screen bg-background p-4 md:p-8">
      {/* Hidden canvas for grayscale conversion */}
      <canvas ref={canvasRef} className="hidden" />
      
      {/* Header */}
      <div className="mx-auto max-w-6xl">
        <div className="mb-8 text-center">
          <div className="mb-4 flex items-center justify-center gap-2">
            <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-primary/20">
              <Sparkles className="h-6 w-6 text-primary" />
            </div>
          </div>
          <h1 className="mb-2 text-3xl font-bold tracking-tight text-foreground md:text-4xl">
            Fashion MNIST Classifier
          </h1>
          <p className="text-muted-foreground">
            Upload a clothing image and let AI identify the category
          </p>
        </div>

        {/* Main Content Grid */}
        <div className="grid gap-6 lg:grid-cols-2">
          {/* Left Panel - Image Upload & Preview */}
          <div className="space-y-6">
            {/* Upload Zone */}
            <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
              <CardHeader className="pb-4">
                <CardTitle className="flex items-center gap-2 text-lg">
                  <Upload className="h-5 w-5 text-primary" />
                  Upload Image
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div
                  onClick={() => fileInputRef.current?.click()}
                  onDrop={handleDrop}
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                  className={`relative cursor-pointer rounded-xl border-2 border-dashed p-8 text-center transition-all duration-300 ${
                    isDragging
                      ? "border-primary bg-primary/10"
                      : "border-border hover:border-primary/50 hover:bg-secondary/50"
                  }`}
                >
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*"
                    onChange={handleFileChange}
                    className="hidden"
                  />
                  <div className="flex flex-col items-center gap-3">
                    <div className="flex h-14 w-14 items-center justify-center rounded-full bg-primary/20">
                      <ImageIcon className="h-7 w-7 text-primary" />
                    </div>
                    <div>
                      <p className="font-medium text-foreground">
                        Drop your image here
                      </p>
                      <p className="text-sm text-muted-foreground">
                        or click to browse
                      </p>
                    </div>
                    <p className="text-xs text-muted-foreground">
                      Supports JPG, PNG, WebP
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Image Previews */}
            {originalImage && (
              <div className="grid gap-4 sm:grid-cols-2">
                {/* Original Image */}
                <Card className="overflow-hidden border-border/50 bg-card/50 backdrop-blur-sm">
                  <CardHeader className="pb-3">
                    <CardTitle className="text-sm font-medium text-muted-foreground">
                      Uploaded Image
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="pt-0">
                    <div className="relative aspect-square overflow-hidden rounded-lg bg-secondary/50">
                      <img
                        src={originalImage}
                        alt="Original uploaded"
                        className="h-full w-full object-contain"
                      />
                    </div>
                  </CardContent>
                </Card>

                {/* Grayscale Image */}
                <Card className="overflow-hidden border-border/50 bg-card/50 backdrop-blur-sm">
                  <CardHeader className="pb-3">
                    <CardTitle className="text-sm font-medium text-muted-foreground">
                      Grayscale (28×28)
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="pt-0">
                    <div className="relative aspect-square overflow-hidden rounded-lg bg-secondary/50">
                      {grayscaleImage && (
                        <img
                          src={grayscaleImage}
                          alt="Grayscale version"
                          className="h-full w-full object-contain"
                          style={{ imageRendering: "pixelated" }}
                        />
                      )}
                    </div>
                  </CardContent>
                </Card>
              </div>
            )}
          </div>

          {/* Right Panel - Classification */}
          <div className="space-y-6">
            {/* Classification Result */}
            <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
              <CardHeader className="pb-4">
                <CardTitle className="flex items-center gap-2 text-lg">
                  <Zap className="h-5 w-5 text-accent" />
                  Predicted Category
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex min-h-[200px] flex-col items-center justify-center rounded-xl bg-secondary/30 p-6">
                  {isClassifying ? (
                    <div className="flex flex-col items-center gap-4">
                      <div className="relative">
                        <div className="h-16 w-16 animate-spin rounded-full border-4 border-primary/30 border-t-primary" />
                        <Sparkles className="absolute left-1/2 top-1/2 h-6 w-6 -translate-x-1/2 -translate-y-1/2 text-primary" />
                      </div>
                      <p className="animate-pulse text-muted-foreground">
                        Analyzing image...
                      </p>
                    </div>
                  ) : prediction ? (
                    <div className="flex flex-col items-center gap-4 text-center">
                      <div className="text-6xl">{LABEL_ICONS[prediction]}</div>
                      <div>
                        <p className="text-3xl font-bold text-foreground">
                          {prediction}
                        </p>
                        {confidence && (
                          <div className="mt-3 flex items-center justify-center gap-2">
                            <div className="h-2 w-32 overflow-hidden rounded-full bg-secondary">
                              <div
                                className="h-full bg-gradient-to-r from-primary to-accent transition-all duration-500"
                                style={{ width: `${confidence}%` }}
                              />
                            </div>
                            <span className="text-sm font-medium text-accent">
                              {confidence.toFixed(1)}%
                            </span>
                          </div>
                        )}
                      </div>
                    </div>
                  ) : (
                    <div className="flex flex-col items-center gap-3 text-center">
                      <div className="flex h-16 w-16 items-center justify-center rounded-full bg-muted">
                        <Sparkles className="h-8 w-8 text-muted-foreground" />
                      </div>
                      <p className="text-muted-foreground">
                        Upload an image and click Classify
                      </p>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>

            {/* Action Buttons */}
            <div className="flex gap-3">
              <Button
                onClick={classifyImage}
                disabled={!grayscaleImage || isClassifying}
                className="flex-1 gap-2 bg-primary text-primary-foreground hover:bg-primary/90"
                size="lg"
              >
                <Sparkles className="h-5 w-5" />
                {isClassifying ? "Classifying..." : "Classify"}
              </Button>
              <Button
                onClick={resetClassifier}
                variant="outline"
                size="lg"
                className="gap-2"
                disabled={!originalImage}
              >
                <RotateCcw className="h-5 w-5" />
                Reset
              </Button>
            </div>

            {/* Categories Reference */}
            <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
              <CardHeader className="pb-3">
                <CardTitle className="text-sm font-medium text-muted-foreground">
                  Supported Categories
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 gap-2 sm:grid-cols-5">
                  {FASHION_LABELS.map((label) => (
                    <div
                      key={label}
                      className={`flex flex-col items-center gap-1 rounded-lg p-2 text-center transition-all ${
                        prediction === label
                          ? "bg-primary/20 ring-1 ring-primary"
                          : "bg-secondary/30"
                      }`}
                    >
                      <span className="text-lg">{LABEL_ICONS[label]}</span>
                      <span className="text-[10px] leading-tight text-muted-foreground">
                        {label}
                      </span>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        </div>

        {/* Footer */}
        <div className="mt-8 text-center">
          <p className="text-xs text-muted-foreground">
            Fashion MNIST Neural Network • 10 Clothing Categories • 28×28 Grayscale Input
          </p>
        </div>
      </div>
    </div>
  )
}
