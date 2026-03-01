package utils

import (
	"image"
	"image/color"
	"math"
)

// ---------------------------------------------------------------------------
// Shared image utilities used by multiple pipeline stages
// ---------------------------------------------------------------------------

var ColorRGBAWhite = color.RGBA{R: 255, G: 255, B: 255, A: 255}

func ColorRGBA(r32, g32, b32, a32 uint32) color.RGBA {
	return color.RGBA{
		R: uint8(r32 >> 8),
		G: uint8(g32 >> 8),
		B: uint8(b32 >> 8),
		A: uint8(a32 >> 8),
	}
}

// BilinearSample returns the bilinear-interpolated RGBA at fractional (x, y).
func BilinearSample(img image.Image, x, y float64) color.RGBA {
	bounds := img.Bounds()
	x0 := int(x)
	y0 := int(y)
	x1 := x0 + 1
	y1 := y0 + 1
	fx := x - float64(x0)
	fy := y - float64(y0)

	x0 = ClampInt(x0, bounds.Min.X, bounds.Max.X-1)
	y0 = ClampInt(y0, bounds.Min.Y, bounds.Max.Y-1)
	x1 = ClampInt(x1, bounds.Min.X, bounds.Max.X-1)
	y1 = ClampInt(y1, bounds.Min.Y, bounds.Max.Y-1)

	r00, g00, b00, _ := img.At(x0, y0).RGBA()
	r10, g10, b10, _ := img.At(x1, y0).RGBA()
	r01, g01, b01, _ := img.At(x0, y1).RGBA()
	r11, g11, b11, _ := img.At(x1, y1).RGBA()

	lerp := func(v00, v10, v01, v11 uint32) uint8 {
		v := float64(v00)*(1-fx)*(1-fy) +
			float64(v10)*fx*(1-fy) +
			float64(v01)*(1-fx)*fy +
			float64(v11)*fx*fy
		return uint8(v / 257.0) // 65535/255 ≈ 257
	}
	return color.RGBA{
		R: lerp(r00, r10, r01, r11),
		G: lerp(g00, g10, g01, g11),
		B: lerp(b00, b10, b01, b11),
		A: 255,
	}
}

// BilinearResize returns a new RGBA image of size (dstW, dstH).
func BilinearResize(src image.Image, dstW, dstH int) *image.RGBA {
	bounds := src.Bounds()
	srcW := bounds.Max.X - bounds.Min.X
	srcH := bounds.Max.Y - bounds.Min.Y
	out := image.NewRGBA(image.Rect(0, 0, dstW, dstH))
	for dy := 0; dy < dstH; dy++ {
		sy := float64(dy)*float64(srcH)/float64(dstH) + float64(bounds.Min.Y)
		for dx := 0; dx < dstW; dx++ {
			sx := float64(dx)*float64(srcW)/float64(dstW) + float64(bounds.Min.X)
			out.SetRGBA(dx, dy, BilinearSample(src, sx, sy))
		}
	}
	return out
}

// ---------------------------------------------------------------------------
// Homography (DLT): 4 point correspondences → 3×3 matrix
// ---------------------------------------------------------------------------

func computeHomography(src, dst [4][2]float64) [9]float64 {
	// Build 8×8 system A·h = b (with h9=1 constraint).
	var A [8][8]float64
	var b [8]float64
	for i := 0; i < 4; i++ {
		sx, sy := src[i][0], src[i][1]
		dx, dy := dst[i][0], dst[i][1]
		A[2*i] = [8]float64{sx, sy, 1, 0, 0, 0, -dx * sx, -dx * sy}
		b[2*i] = dx
		A[2*i+1] = [8]float64{0, 0, 0, sx, sy, 1, -dy * sx, -dy * sy}
		b[2*i+1] = dy
	}
	h := gaussianElim8(A, b)
	return [9]float64{h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1}
}

func gaussianElim8(A [8][8]float64, b [8]float64) [8]float64 {
	// Augmented matrix.
	var mat [8][9]float64
	for i := 0; i < 8; i++ {
		copy(mat[i][:8], A[i][:])
		mat[i][8] = b[i]
	}
	// Forward elimination with partial pivoting.
	for col := 0; col < 8; col++ {
		pivot := col
		for row := col + 1; row < 8; row++ {
			if math.Abs(mat[row][col]) > math.Abs(mat[pivot][col]) {
				pivot = row
			}
		}
		mat[col], mat[pivot] = mat[pivot], mat[col]
		if math.Abs(mat[col][col]) < 1e-12 {
			continue
		}
		for row := col + 1; row < 8; row++ {
			f := mat[row][col] / mat[col][col]
			for j := col; j <= 8; j++ {
				mat[row][j] -= f * mat[col][j]
			}
		}
	}
	// Back substitution.
	var x [8]float64
	for i := 7; i >= 0; i-- {
		x[i] = mat[i][8]
		for j := i + 1; j < 8; j++ {
			x[i] -= mat[i][j] * x[j]
		}
		if math.Abs(mat[i][i]) > 1e-12 {
			x[i] /= mat[i][i]
		}
	}
	return x
}

// invertH computes the inverse of a 3×3 homogeneous matrix.
func invertH(H [9]float64) [9]float64 {
	a, b, c := H[0], H[1], H[2]
	d, e, f := H[3], H[4], H[5]
	g, h, k := H[6], H[7], H[8]
	det := a*(e*k-f*h) - b*(d*k-f*g) + c*(d*h-e*g)
	if math.Abs(det) < 1e-12 {
		return H
	}
	id := 1.0 / det
	return [9]float64{
		id * (e*k - f*h), id * (c*h - b*k), id * (b*f - c*e),
		id * (f*g - d*k), id * (a*k - c*g), id * (c*d - a*f),
		id * (d*h - e*g), id * (b*g - a*h), id * (a*e - b*d),
	}
}

// ---------------------------------------------------------------------------
// NCHW tensor builder with per-channel normalisation
// ---------------------------------------------------------------------------

func ImageToNCHW(img *image.RGBA, H, W int, mean, std [3]float64) []float32 {
	data := make([]float32, 3*H*W)
	bounds := img.Bounds()
	for y := 0; y < H; y++ {
		iy := y + bounds.Min.Y
		for x := 0; x < W; x++ {
			ix := x + bounds.Min.X
			pix := img.RGBAAt(ix, iy)
			rv := float64(pix.R) / 255.0
			gv := float64(pix.G) / 255.0
			bv := float64(pix.B) / 255.0
			idx := y*W + x
			data[0*H*W+idx] = float32((rv - mean[0]) / std[0])
			data[1*H*W+idx] = float32((gv - mean[1]) / std[1])
			data[2*H*W+idx] = float32((bv - mean[2]) / std[2])
		}
	}
	return data
}

// ImageToNCHWFromImage is a slower variant for non-RGBA images.
func ImageToNCHWFromImage(img image.Image, H, W int, mean, std [3]float64) []float32 {
	data := make([]float32, 3*H*W)
	bounds := img.Bounds()
	for y := 0; y < H; y++ {
		iy := y + bounds.Min.Y
		for x := 0; x < W; x++ {
			ix := x + bounds.Min.X
			r32, g32, b32, _ := img.At(ix, iy).RGBA()
			rv := float64(r32) / 65535.0
			gv := float64(g32) / 65535.0
			bv := float64(b32) / 65535.0
			idx := y*W + x
			data[0*H*W+idx] = float32((rv - mean[0]) / std[0])
			data[1*H*W+idx] = float32((gv - mean[1]) / std[1])
			data[2*H*W+idx] = float32((bv - mean[2]) / std[2])
		}
	}
	return data
}

// Rotate90CCW rotates an *image.RGBA 90° counter-clockwise.
// Input (srcW × srcH) → Output (srcH × srcW).
// Top of the input maps to the left of the output, so vertical text that
// reads top-to-bottom becomes horizontal text reading left-to-right.
func Rotate90CCW(img *image.RGBA) *image.RGBA {
	bounds := img.Bounds()
	srcW := bounds.Dx() // columns of input
	srcH := bounds.Dy() // rows of input
	// After 90° CCW: output width = srcH, output height = srcW.
	out := image.NewRGBA(image.Rect(0, 0, srcH, srcW))
	for ox := 0; ox < srcH; ox++ { // output column = input row
		for oy := 0; oy < srcW; oy++ { // output row   = input (reversed) column
			ix := srcW - 1 - oy
			iy := ox
			out.SetRGBA(ox, oy, img.RGBAAt(bounds.Min.X+ix, bounds.Min.Y+iy))
		}
	}
	return out
}

// ---------------------------------------------------------------------------
// Perspective warp: maps the quad to a (dstW × dstH) rectangle.
// ---------------------------------------------------------------------------

// FloatQuad converts the integer quad to float64 points.
func FloatQuad(q [4][2]int) [4][2]float64 {
	return [4][2]float64{
		{float64(q[0][0]), float64(q[0][1])},
		{float64(q[1][0]), float64(q[1][1])},
		{float64(q[2][0]), float64(q[2][1])},
		{float64(q[3][0]), float64(q[3][1])},
	}
}

// PerspectiveWarp crops the img quad into a (dstW × dstH) RGBA image.
// src must be ordered [top-left, top-right, bottom-right, bottom-left].
func PerspectiveWarp(img image.Image, src [4][2]float64, dstW, dstH int) *image.RGBA {
	if dstW <= 0 || dstH <= 0 {
		return nil
	}
	// Destination rectangle corners.
	dst := [4][2]float64{
		{0, 0},
		{float64(dstW - 1), 0},
		{float64(dstW - 1), float64(dstH - 1)},
		{0, float64(dstH - 1)},
	}

	H := computeHomography(src, dst)
	if H == [9]float64{} {
		return nil
	}
	hInv := invertH(H)

	out := image.NewRGBA(image.Rect(0, 0, dstW, dstH))
	bounds := img.Bounds()

	for dy := 0; dy < dstH; dy++ {
		for dx := 0; dx < dstW; dx++ {
			// Apply inverse homography: (dx,dy) → (sx,sy)
			xf, yf := float64(dx), float64(dy)
			w := hInv[6]*xf + hInv[7]*yf + hInv[8]
			if math.Abs(w) < 1e-10 {
				out.SetRGBA(dx, dy, ColorRGBAWhite)
				continue
			}
			sx := (hInv[0]*xf + hInv[1]*yf + hInv[2]) / w
			sy := (hInv[3]*xf + hInv[4]*yf + hInv[5]) / w

			if sx < float64(bounds.Min.X) || sy < float64(bounds.Min.Y) ||
				sx > float64(bounds.Max.X-1) || sy > float64(bounds.Max.Y-1) {
				out.SetRGBA(dx, dy, ColorRGBAWhite)
				continue
			}
			out.SetRGBA(dx, dy, BilinearSample(img, sx, sy))
		}
	}
	return out
}

// ---------------------------------------------------------------------------
// Order 4 points: [top-left, top-right, bottom-right, bottom-left]
// ---------------------------------------------------------------------------

func OrderPoints4(pts [4][2]float64) [4][2]float64 {
	// In image coords (y-down):
	//   TL = min(x+y),  BR = max(x+y)
	//   TR = max(x-y) — large x, small y
	//   BL = min(x-y) — small x, large y
	sums := [4]float64{pts[0][0] + pts[0][1], pts[1][0] + pts[1][1], pts[2][0] + pts[2][1], pts[3][0] + pts[3][1]}
	diffs := [4]float64{pts[0][0] - pts[0][1], pts[1][0] - pts[1][1], pts[2][0] - pts[2][1], pts[3][0] - pts[3][1]}

	tlIdx, brIdx, trIdx, blIdx := 0, 0, 0, 0
	for i := 1; i < 4; i++ {
		if sums[i] < sums[tlIdx] {
			tlIdx = i
		}
		if sums[i] > sums[brIdx] {
			brIdx = i
		}
		if diffs[i] > diffs[trIdx] { // max(x-y) → TR
			trIdx = i
		}
		if diffs[i] < diffs[blIdx] { // min(x-y) → BL
			blIdx = i
		}
	}
	return [4][2]float64{pts[tlIdx], pts[trIdx], pts[brIdx], pts[blIdx]}
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

func PointDistance(a, b [2]float64) float64 {
	dx, dy := a[0]-b[0], a[1]-b[1]
	return math.Sqrt(dx*dx + dy*dy)
}
