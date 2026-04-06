"use client"

import { useState, useRef, useCallback } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Upload, Sparkles, ImageIcon, Zap, RotateCcw, AlertCircle, Crop, ChevronDown, ChevronUp, CheckCircle2, XCircle, TriangleAlert } from "lucide-react"
import { useIsMobile } from "@/hooks/use-mobile"

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? ""

const FASHION_LABELS = [
  "T-shirt/top",
  "Trouser",
  "Pullover",
  "Dress",
  "Coat",
  "Sandal",
  "Shirt",
  "Sneaker",
  "Bag",
  "Ankle boot",
]

const LABEL_ICONS: Record<string, string> = {
  "T-shirt/top": "👕",
  "Trouser":     "👖",
  "Pullover":    "🧥",
  "Dress":       "👗",
  "Coat":        "🧥",
  "Sandal":      "🩴",
  "Shirt":       "👔",
  "Sneaker":     "👟",
  "Bag":         "👜",
  "Ankle boot":  "🥾",
}

interface TopProb {
  label: string
  probability: number
}

interface ClassifyResponse {
  label: string
  confidence: number
  top_probs: TopProb[]
}

// Crop rect in image-relative pixels
interface CropRect { x: number; y: number; w: number; h: number }

type DragMode =
  | { type: "move" }
  | { type: "resize"; corner: "tl" | "tr" | "bl" | "br" }
  | null

// ─── helpers ────────────────────────────────────────────────────────────────

function clamp(v: number, lo: number, hi: number) {
  return Math.max(lo, Math.min(hi, v))
}

/** Convert a CropRect in image-pixel space to CSS % on the preview container */
function rectToCss(crop: CropRect, imgW: number, imgH: number) {
  return {
    left:   `${(crop.x / imgW) * 100}%`,
    top:    `${(crop.y / imgH) * 100}%`,
    width:  `${(crop.w / imgW) * 100}%`,
    height: `${(crop.h / imgH) * 100}%`,
  }
}

/** Default crop: centered 70% of image */
function defaultCrop(w: number, h: number): CropRect {
  const cw = Math.round(w * 0.7)
  const ch = Math.round(h * 0.7)
  return { x: Math.round((w - cw) / 2), y: Math.round((h - ch) / 2), w: cw, h: ch }
}

/** Apply crop to an offscreen canvas and return its data-URL */
function applyCrop(src: string, crop: CropRect, naturalW: number, naturalH: number): Promise<string> {
  return new Promise((resolve) => {
    const img = new Image()
    img.onload = () => {
      const off = document.createElement("canvas")
      off.width  = crop.w
      off.height = crop.h
      const ctx = off.getContext("2d")!
      ctx.drawImage(img, crop.x, crop.y, crop.w, crop.h, 0, 0, crop.w, crop.h)
      resolve(off.toDataURL("image/png"))
    }
    img.src = src
  })
}

// ─── component ──────────────────────────────────────────────────────────────

export function FashionClassifier() {
  const isMobile = useIsMobile()
  const handleSize = isMobile ? 24 : 12

  // ── existing state ──────────────────────────────────────────────────────
  const [originalImage, setOriginalImage]   = useState<string | null>(null)
  const [grayscaleImage, setGrayscaleImage] = useState<string | null>(null)
  const [uploadedFile, setUploadedFile]     = useState<File | null>(null)
  const [prediction, setPrediction]         = useState<string | null>(null)
  const [confidence, setConfidence]         = useState<number | null>(null)
  const [topProbs, setTopProbs]             = useState<TopProb[]>([])
  const [isClassifying, setIsClassifying]   = useState(false)
  const [isDragging, setIsDragging]         = useState(false)
  const [error, setError]                   = useState<string | null>(null)

  // ── guidance state ───────────────────────────────────────────────────────
  const [tipsOpen, setTipsOpen] = useState(false)

  // ── crop state ──────────────────────────────────────────────────────────
  const [cropActive, setCropActive] = useState(false)
  const [cropRect, setCropRect]     = useState<CropRect>({ x: 0, y: 0, w: 0, h: 0 })
  // natural dimensions of the loaded image
  const [naturalSize, setNaturalSize] = useState({ w: 0, h: 0 })

  // ── refs ────────────────────────────────────────────────────────────────
  const fileInputRef   = useRef<HTMLInputElement>(null)
  const canvasRef      = useRef<HTMLCanvasElement>(null)
  const previewRef     = useRef<HTMLDivElement>(null)   // the image preview container
  const imgRef         = useRef<HTMLImageElement>(null) // the visible preview <img>

  // drag state stored in refs to avoid re-render on every mousemove
  const dragMode    = useRef<DragMode>(null)
  const dragStart   = useRef({ px: 0, py: 0, rect: { x: 0, y: 0, w: 0, h: 0 } })

  // ── canvas pipeline (unchanged) ─────────────────────────────────────────
  const convertToGrayscale = useCallback((imageSrc: string): Promise<string> => {
    return new Promise((resolve) => {
      const img = new Image()
      img.crossOrigin = "anonymous"
      img.onload = () => {
        const canvas = canvasRef.current
        if (!canvas) return
        const ctx = canvas.getContext("2d")
        if (!ctx) return
        canvas.width  = 28
        canvas.height = 28
        ctx.drawImage(img, 0, 0, 28, 28)
        const imageData = ctx.getImageData(0, 0, 28, 28)
        const data = imageData.data
        for (let i = 0; i < data.length; i += 4) {
          const gray = data[i] * 0.299 + data[i + 1] * 0.587 + data[i + 2] * 0.114
          data[i] = gray; data[i + 1] = gray; data[i + 2] = gray
        }
        ctx.putImageData(imageData, 0, 0)
        resolve(canvas.toDataURL())
      }
      img.src = imageSrc
    })
  }, [])

  // ── image upload ─────────────────────────────────────────────────────────
  const handleImageUpload = useCallback((file: File) => {
    setError(null)
    setPrediction(null)
    setConfidence(null)
    setTopProbs([])
    setGrayscaleImage(null)
    setUploadedFile(file)
    setCropActive(false)

    const reader = new FileReader()
    reader.onload = (e) => {
      const src = e.target?.result as string
      setOriginalImage(src)

      // measure natural dimensions then activate crop
      const probe = new Image()
      probe.onload = () => {
        const nw = probe.naturalWidth
        const nh = probe.naturalHeight
        setNaturalSize({ w: nw, h: nh })
        setCropRect(defaultCrop(nw, nh))
        setCropActive(true)
      }
      probe.src = src
    }
    reader.readAsDataURL(file)
  }, [])

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) handleImageUpload(file)
  }

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    const file = e.dataTransfer.files[0]
    if (file && file.type.startsWith("image/")) handleImageUpload(file)
  }, [handleImageUpload])

  const handleDragOver  = (e: React.DragEvent) => { e.preventDefault(); setIsDragging(true) }
  const handleDragLeave = () => setIsDragging(false)

  // ── pointer → image-pixel coordinate conversion ──────────────────────────
  const pointerToImagePx = useCallback((clientX: number, clientY: number) => {
    const el = previewRef.current
    if (!el) return { px: 0, py: 0 }
    const rect = el.getBoundingClientRect()
    // The <img> uses object-contain inside a square container.
    // We need to map from container coords to image-natural coords.
    const containerW = rect.width
    const containerH = rect.height
    const { w: nw, h: nh } = naturalSize
    if (!nw || !nh) return { px: 0, py: 0 }

    const scale    = Math.min(containerW / nw, containerH / nh)
    const renderedW = nw * scale
    const renderedH = nh * scale
    const offsetX   = (containerW - renderedW) / 2
    const offsetY   = (containerH - renderedH) / 2

    const px = (clientX - rect.left - offsetX) / scale
    const py = (clientY - rect.top  - offsetY) / scale
    return { px, py }
  }, [naturalSize])

  // ── drag handlers ─────────────────────────────────────────────────────────
  const onPointerDown = useCallback((
    e: React.PointerEvent,
    mode: DragMode,
  ) => {
    e.stopPropagation()
    e.currentTarget.setPointerCapture(e.pointerId)
    const { px, py } = pointerToImagePx(e.clientX, e.clientY)
    dragMode.current  = mode
    dragStart.current = { px, py, rect: { ...cropRect } }
  }, [cropRect, pointerToImagePx])

  const onPointerMove = useCallback((e: React.PointerEvent) => {
    if (!dragMode.current) return
    const { w: nw, h: nh } = naturalSize
    const { px: startPx, py: startPy, rect: startRect } = dragStart.current
    const { px, py } = pointerToImagePx(e.clientX, e.clientY)
    const dx = px - startPx
    const dy = py - startPy
    const MIN = 20

    setCropRect((prev) => {
      let { x, y, w, h } = startRect

      if (dragMode.current?.type === "move") {
        x = clamp(x + dx, 0, nw - w)
        y = clamp(y + dy, 0, nh - h)
      } else if (dragMode.current?.type === "resize") {
        const corner = dragMode.current.corner
        if (corner === "tl") {
          const nx = clamp(x + dx, 0, x + w - MIN)
          const ny = clamp(y + dy, 0, y + h - MIN)
          w = w + (x - nx); h = h + (y - ny); x = nx; y = ny
        } else if (corner === "tr") {
          const ny = clamp(y + dy, 0, y + h - MIN)
          w = clamp(w + dx, MIN, nw - x)
          h = h + (y - ny); y = ny
        } else if (corner === "bl") {
          const nx = clamp(x + dx, 0, x + w - MIN)
          w = w + (x - nx); x = nx
          h = clamp(h + dy, MIN, nh - y)
        } else if (corner === "br") {
          w = clamp(w + dx, MIN, nw - x)
          h = clamp(h + dy, MIN, nh - y)
        }
      }

      return { x: Math.round(x), y: Math.round(y), w: Math.round(w), h: Math.round(h) }
    })
  }, [naturalSize, pointerToImagePx])

  const onPointerUp = useCallback(() => {
    dragMode.current = null
  }, [])

  // ── classify helpers ──────────────────────────────────────────────────────
  const runClassify = useCallback(async (imageSrc: string) => {
    setIsClassifying(true)
    setError(null)
    setPrediction(null)
    setConfidence(null)
    setTopProbs([])

    try {
      // feed imageSrc through existing 28×28 grayscale canvas pipeline
      const grayscale = await convertToGrayscale(imageSrc)
      setGrayscaleImage(grayscale)

      // canvas.toBlob → FormData → POST
      const canvas = canvasRef.current!
      const blob = await new Promise<Blob>((res) =>
        canvas.toBlob((b) => res(b!), "image/png")
      )
      const formData = new FormData()
      formData.append("image", blob, "image.png")

      const controller = new AbortController()
      const timeoutId  = setTimeout(() => controller.abort(), 60000)
      const response   = await fetch(`${API_BASE}/api/classify`, {
        method: "POST",
        body: formData,
        signal: controller.signal,
      })
      clearTimeout(timeoutId)

      if (!response.ok) {
        const errBody = await response.json().catch(() => ({ detail: "Unknown error" }))
        throw new Error(errBody.detail ?? `Server error ${response.status}`)
      }

      const data: ClassifyResponse = await response.json()
      setPrediction(data.label)
      setConfidence(data.confidence)
      setTopProbs(data.top_probs ?? [])
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") {
        setError("Request timed out. The backend may be slow — Render free tier takes ~30s to wake up. Try again.")
      } else if (err instanceof TypeError && err.message.toLowerCase().includes("fetch")) {
        setError("Cannot reach the backend. CORS or network error.")
      } else {
        setError(err instanceof Error ? err.message : `Classification failed. API: ${API_BASE || "(not set)"}`)
      }
    } finally {
      setIsClassifying(false)
    }
  }, [convertToGrayscale])

  const handleCropAndClassify = useCallback(async () => {
    if (!originalImage) return
    setCropActive(false)
    const cropped = await applyCrop(originalImage, cropRect, naturalSize.w, naturalSize.h)
    await runClassify(cropped)
  }, [originalImage, cropRect, naturalSize, runClassify])

  const handleSkipCrop = useCallback(async () => {
    if (!originalImage) return
    setCropActive(false)
    await runClassify(originalImage)
  }, [originalImage, runClassify])

  // kept for the Reset button — same as before
  const classifyImage = handleSkipCrop

  const resetClassifier = () => {
    setOriginalImage(null)
    setGrayscaleImage(null)
    setUploadedFile(null)
    setPrediction(null)
    setConfidence(null)
    setTopProbs([])
    setError(null)
    setCropActive(false)
    setCropRect({ x: 0, y: 0, w: 0, h: 0 })
    if (fileInputRef.current) fileInputRef.current.value = ""
  }

  const sortedProbs = [...topProbs].sort((a, b) => b.probability - a.probability)

  // ── class-specific warning sets ──────────────────────────────────────────
  const TOPS_LABELS  = new Set(["Shirt", "T-shirt/top", "Pullover", "Coat"])
  const SHOES_LABELS = new Set(["Sandal", "Sneaker", "Ankle boot"])

  // ── guidance card (shared between empty state and collapsible reminder) ──
  const guidanceBody = (
    <div className="space-y-3 text-xs">
      <div className="space-y-1">
        <p className="flex items-center gap-1 font-medium text-emerald-400">
          <CheckCircle2 className="h-3.5 w-3.5 shrink-0" /> Do
        </p>
        <ul className="space-y-0.5 pl-5 text-muted-foreground">
          <li>Lay the garment flat on a plain surface</li>
          <li>Photograph from directly above, centred in frame</li>
          <li>Use a plain white, black, or grey background</li>
          <li>One item only per photo</li>
        </ul>
      </div>
      <div className="space-y-1">
        <p className="flex items-center gap-1 font-medium text-destructive">
          <XCircle className="h-3.5 w-3.5 shrink-0" /> Avoid
        </p>
        <ul className="space-y-0.5 pl-5 text-muted-foreground">
          <li>Garments being worn on a person</li>
          <li>Shoes photographed on feet</li>
          <li>Cluttered backgrounds or other items in frame</li>
          <li>Extreme angles or partial garments</li>
        </ul>
      </div>
      <p className="text-[10px] text-muted-foreground/70 pt-1 border-t border-border/40">
        The model was trained on studio product photos. Images that match this style predict most accurately.
      </p>
    </div>
  )

  // CSS for the crop overlay handles
  const hSize = `${handleSize}px`
  const hOff  = `${-handleSize / 2}px`

  return (
    <div className="min-h-screen bg-background p-4 md:p-8">
      <canvas ref={canvasRef} className="hidden" />

      <div className="mx-auto max-w-6xl">
        {/* Header */}
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

        <div className="grid gap-6 lg:grid-cols-2">
          {/* ── Left Panel ── */}
          <div className="space-y-6">
            {/* Upload zone */}
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
                      <p className="font-medium text-foreground">Drop your image here</p>
                      <p className="text-sm text-muted-foreground">or click to browse</p>
                    </div>
                    <p className="text-xs text-muted-foreground">Supports JPG, PNG, WebP</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Guidance card — full version, shown only before upload */}
            {!originalImage && (
              <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
                <CardHeader className="pb-3">
                  <CardTitle className="flex items-center gap-2 text-sm font-medium text-foreground">
                    <CheckCircle2 className="h-4 w-4 text-emerald-400" />
                    For best predictions
                  </CardTitle>
                </CardHeader>
                <CardContent>{guidanceBody}</CardContent>
              </Card>
            )}

            {/* Image preview + crop overlay */}
            {originalImage && (
              <div className="space-y-3">
                <div className="grid gap-4 sm:grid-cols-2">
                  {/* Uploaded image with crop overlay */}
                  <Card className="overflow-hidden border-border/50 bg-card/50 backdrop-blur-sm">
                    <CardHeader className="pb-3">
                      <CardTitle className="flex items-center gap-2 text-sm font-medium text-muted-foreground">
                        {cropActive
                          ? <><Crop className="h-3.5 w-3.5" /> Crop Selection</>
                          : "Uploaded Image"
                        }
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="pt-0">
                      {/* Container — pointer events for drag */}
                      <div
                        ref={previewRef}
                        className="relative aspect-square overflow-hidden rounded-lg bg-secondary/50 touch-none"
                        onPointerMove={cropActive ? onPointerMove : undefined}
                        onPointerUp={cropActive ? onPointerUp : undefined}
                      >
                        <img
                          ref={imgRef}
                          src={originalImage}
                          alt="Original uploaded"
                          className="h-full w-full object-contain"
                          draggable={false}
                        />

                        {/* ── Crop overlay ── */}
                        {cropActive && naturalSize.w > 0 && (() => {
                          const css = rectToCss(cropRect, naturalSize.w, naturalSize.h)
                          return (
                            <>
                              {/* Dark mask — 4 rects around the selection */}
                              {/* top */}
                              <div className="pointer-events-none absolute inset-x-0 top-0 bg-black/50"
                                style={{ height: css.top }} />
                              {/* bottom */}
                              <div className="pointer-events-none absolute inset-x-0 bottom-0 bg-black/50"
                                style={{ top: `calc(${css.top} + ${css.height})` }} />
                              {/* left */}
                              <div className="pointer-events-none absolute bg-black/50"
                                style={{ top: css.top, height: css.height, left: 0, width: css.left }} />
                              {/* right */}
                              <div className="pointer-events-none absolute bg-black/50"
                                style={{ top: css.top, height: css.height, right: 0, left: `calc(${css.left} + ${css.width})` }} />

                              {/* Selection box — draggable interior */}
                              <div
                                className="absolute cursor-move border-2 border-primary"
                                style={{ ...css, boxSizing: "border-box" }}
                                onPointerDown={(e) => onPointerDown(e, { type: "move" })}
                              >
                                {/* Dimension label */}
                                <span className="pointer-events-none absolute bottom-1 left-1/2 -translate-x-1/2 rounded bg-black/70 px-1 py-0.5 text-[9px] text-white whitespace-nowrap">
                                  {cropRect.w} × {cropRect.h} px
                                </span>

                                {/* Corner handles */}
                                {(["tl","tr","bl","br"] as const).map((corner) => (
                                  <div
                                    key={corner}
                                    className="absolute rounded-sm bg-primary border border-background"
                                    style={{
                                      width: hSize, height: hSize,
                                      top:    corner.startsWith("t") ? hOff : "auto",
                                      bottom: corner.startsWith("b") ? hOff : "auto",
                                      left:   corner.endsWith("l")   ? hOff : "auto",
                                      right:  corner.endsWith("r")   ? hOff : "auto",
                                      cursor: corner === "tl" || corner === "br" ? "nwse-resize" : "nesw-resize",
                                    }}
                                    onPointerDown={(e) => onPointerDown(e, { type: "resize", corner })}
                                  />
                                ))}
                              </div>
                            </>
                          )
                        })()}
                      </div>
                    </CardContent>
                  </Card>

                  {/* Grayscale preview */}
                  <Card className="overflow-hidden border-border/50 bg-card/50 backdrop-blur-sm">
                    <CardHeader className="pb-3">
                      <CardTitle className="text-sm font-medium text-muted-foreground">
                        Grayscale (28×28)
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="pt-0">
                      <div className="relative aspect-square overflow-hidden rounded-lg bg-secondary/50">
                        {grayscaleImage ? (
                          <img
                            src={grayscaleImage}
                            alt="Grayscale version"
                            className="h-full w-full object-contain"
                            style={{ imageRendering: "pixelated" }}
                          />
                        ) : (
                          <div className="flex h-full items-center justify-center text-xs text-muted-foreground">
                            {cropActive ? "Crop then classify" : "—"}
                          </div>
                        )}
                      </div>
                    </CardContent>
                  </Card>
                </div>

                {/* Collapsible photo tips reminder — shown after upload, above crop buttons */}
                {cropActive && (
                  <div className="rounded-lg border border-border/50 bg-card/30 overflow-hidden">
                    <button
                      onClick={() => setTipsOpen((o) => !o)}
                      className="flex w-full items-center justify-between px-3 py-2 text-xs text-muted-foreground hover:text-foreground transition-colors"
                    >
                      <span className="flex items-center gap-1.5">
                        <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400" />
                        Photo tips for better accuracy
                      </span>
                      {tipsOpen
                        ? <ChevronUp className="h-3.5 w-3.5" />
                        : <ChevronDown className="h-3.5 w-3.5" />
                      }
                    </button>
                    {tipsOpen && (
                      <div className="px-3 pb-3 border-t border-border/40">
                        <div className="pt-2">{guidanceBody}</div>
                      </div>
                    )}
                  </div>
                )}

                {/* Crop action buttons */}
                {cropActive && (
                  <div className="space-y-2">
                    <div className="flex gap-3">
                      <Button
                        onClick={handleCropAndClassify}
                        className="flex-1 gap-2 bg-primary text-primary-foreground hover:bg-primary/90"
                        size="lg"
                      >
                        <Crop className="h-4 w-4" />
                        Crop &amp; Classify
                      </Button>
                      <Button
                        onClick={handleSkipCrop}
                        variant="ghost"
                        size="lg"
                        className="gap-2"
                      >
                        Skip crop
                      </Button>
                    </div>
                    <p className="text-xs text-muted-foreground">
                      Crop to just the garment for best results. The model was trained on
                      isolated clothing items with plain backgrounds.
                    </p>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* ── Right Panel ── */}
          <div className="space-y-6">
            {/* Error Alert */}
            {error && (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertTitle>Classification failed</AlertTitle>
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

            {/* Prediction Result */}
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
                      <p className="animate-pulse text-muted-foreground">Analyzing image...</p>
                    </div>
                  ) : prediction ? (
                    <div className="flex flex-col items-center gap-4 text-center">
                      <div className="text-6xl">{LABEL_ICONS[prediction]}</div>
                      <div>
                        <p className="text-3xl font-bold text-foreground">{prediction}</p>
                        {confidence !== null && (
                          <div className="mt-3 flex items-center justify-center gap-2">
                            <div className="h-2 w-32 overflow-hidden rounded-full bg-secondary">
                              <div
                                className="h-full bg-gradient-to-r from-primary to-accent transition-all duration-500"
                                style={{ width: `${confidence * 100}%` }}
                              />
                            </div>
                            <span className="text-sm font-medium text-accent">
                              {(confidence * 100).toFixed(1)}%
                            </span>
                          </div>
                        )}
                      </div>
                      {/* Class-specific warning badges */}
                      {TOPS_LABELS.has(prediction) && (
                        <div className="flex items-start gap-2 rounded-lg bg-amber-500/10 border border-amber-500/30 px-3 py-2 text-left text-xs text-amber-400 max-w-xs">
                          <TriangleAlert className="h-3.5 w-3.5 shrink-0 mt-0.5" />
                          <span>These styles look similar at 28×28 pixels — check the top-3 predictions below for alternatives.</span>
                        </div>
                      )}
                      {SHOES_LABELS.has(prediction) && (
                        <div className="flex items-start gap-2 rounded-lg bg-amber-500/10 border border-amber-500/30 px-3 py-2 text-left text-xs text-amber-400 max-w-xs">
                          <TriangleAlert className="h-3.5 w-3.5 shrink-0 mt-0.5" />
                          <span>For shoes: photograph the item laid flat from above, not worn on a foot.</span>
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="flex flex-col items-center gap-3 text-center">
                      <div className="flex h-16 w-16 items-center justify-center rounded-full bg-muted">
                        <Sparkles className="h-8 w-8 text-muted-foreground" />
                      </div>
                      <p className="text-muted-foreground">
                        {cropActive ? "Crop the garment then classify" : "Upload an image and click Classify"}
                      </p>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>

            {/* Action Buttons — shown after crop is dismissed */}
            {!cropActive && originalImage && (
              <div className="flex gap-3">
                <Button
                  onClick={classifyImage}
                  disabled={!uploadedFile || isClassifying}
                  className="flex-1 gap-2 bg-primary text-primary-foreground hover:bg-primary/90"
                  size="lg"
                >
                  <Sparkles className="h-5 w-5" />
                  {isClassifying ? "Classifying..." : "Classify again"}
                </Button>
                <Button
                  onClick={resetClassifier}
                  variant="outline"
                  size="lg"
                  className="gap-2"
                >
                  <RotateCcw className="h-5 w-5" />
                  Reset
                </Button>
              </div>
            )}

            {/* Reset only — shown during crop phase */}
            {cropActive && (
              <div className="flex justify-end">
                <Button onClick={resetClassifier} variant="outline" size="sm" className="gap-2">
                  <RotateCcw className="h-4 w-4" />
                  Reset
                </Button>
              </div>
            )}

            {/* Top Probabilities Breakdown */}
            {sortedProbs.length > 0 && (
              <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
                <CardHeader className="pb-3">
                  <CardTitle className="text-sm font-medium text-muted-foreground">
                    All Category Probabilities
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-2">
                  {sortedProbs.map((item) => (
                    <div key={item.label} className="space-y-1">
                      <div className="flex items-center justify-between text-xs">
                        <span className={`flex items-center gap-1 ${item.label === prediction ? "font-semibold text-primary" : "text-muted-foreground"}`}>
                          {LABEL_ICONS[item.label]} {item.label}
                        </span>
                        <span className={item.label === prediction ? "font-semibold text-primary" : "text-muted-foreground"}>
                          {(item.probability * 100).toFixed(1)}%
                        </span>
                      </div>
                      <Progress
                        value={item.probability * 100}
                        className={item.label === prediction ? "h-1.5 [&>div]:bg-primary" : "h-1.5 [&>div]:bg-muted-foreground/40"}
                      />
                    </div>
                  ))}
                </CardContent>
              </Card>
            )}

            {/* Categories Reference — shown when no results yet */}
            {sortedProbs.length === 0 && (
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
                        className="flex flex-col items-center gap-1 rounded-lg bg-secondary/30 p-2 text-center"
                      >
                        <span className="text-lg">{LABEL_ICONS[label]}</span>
                        <span className="text-[10px] leading-tight text-muted-foreground">{label}</span>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}
          </div>
        </div>

        <div className="mt-8 text-center">
          <p className="text-xs text-muted-foreground">
            Fashion MNIST Neural Network • 10 Clothing Categories • 28×28 Grayscale Input
          </p>
        </div>
      </div>
    </div>
  )
}
