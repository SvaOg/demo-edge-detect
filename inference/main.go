package main

import (
	"fmt"
	"image"
	_ "image/jpeg" // Register JPEG decoder
	_ "image/png"  // Register PNG decoder
	"os"
	"path/filepath"

	ort "github.com/yalue/onnxruntime_go"
	"golang.org/x/image/draw"
)

// Model constants
const (
	ModelInputSize = 640
	ModelChannels  = 3
)

type Detection struct {
	X      float32
	Y      float32
	Width  float32
	Height float32
	Conf   float32
}

func main() {
	// --- 1. Initialization ---
	dllPath, _ := filepath.Abs("onnxruntime.dll")
	ort.SetSharedLibraryPath(dllPath)

	err := ort.InitializeEnvironment()
	if err != nil {
		fmt.Printf("Error initializing environment: %v\n", err)
		return
	}
	defer ort.DestroyEnvironment()

	// --- 2. Model Loading ---
	modelPath := filepath.Join("..", "models", "model.onnx")
	absModelPath, _ := filepath.Abs(modelPath)

	if _, err := os.Stat(absModelPath); os.IsNotExist(err) {
		fmt.Printf("Model file not found: %s\n", absModelPath)
		return
	}

	session, err := ort.NewDynamicAdvancedSession(
		absModelPath,
		[]string{"images"},
		[]string{"output0"},
		nil,
	)
	if err != nil {
		fmt.Printf("Error creating session: %v\n", err)
		return
	}
	defer session.Destroy()
	fmt.Println("Model loaded successfully!")

	// --- 3. Image Loading and Processing ---
	imagePath := filepath.Join("..", "data", "test", "test001.JPEG")
	fmt.Printf("Processing image: %s\n", imagePath)

	inputData, err := loadAndPreprocess(imagePath)
	if err != nil {
		fmt.Printf("Error processing image: %v\n", err)
		// If the image is not there, we don't crash, we exit
		return
	}

	// --- 4. Tensor Preparation ---
	// Shape: [Batch=1, Channels=3, Height=640, Width=640]
	inputShape := ort.NewShape(1, int64(ModelChannels), int64(ModelInputSize), int64(ModelInputSize))
	inputTensor, err := ort.NewTensor(inputShape, inputData)
	if err != nil {
		fmt.Printf("Error creating input tensor: %v\n", err)
		return
	}
	defer inputTensor.Destroy()

	// Output Shape: [1, 5, 8400]
	outputSize := 1 * 5 * 8400
	outputData := make([]float32, outputSize)
	outputShape := ort.NewShape(1, 5, 8400)

	outputTensor, err := ort.NewTensor(outputShape, outputData)
	if err != nil {
		fmt.Printf("Error creating output tensor: %v\n", err)
		return
	}
	defer outputTensor.Destroy()

	// --- 5. Run (Inference) ---
	fmt.Println("Running inference...")
	err = session.Run(
		[]ort.Value{inputTensor},
		[]ort.Value{outputTensor},
	)
	if err != nil {
		fmt.Printf("Error running inference: %v\n", err)
		return
	}

	// --- 6. Post-processing (Parsing YOLO output) ---
	// YOLO output layout: [1, 5, 8400] -> Flattened to 42000
	// Memory structure: [8400 x coords] [8400 y coords] [8400 widths] [8400 heights] [8400 scores]

	numAnchors := 8400
	var bestDetect Detection
	bestDetect.Conf = 0.0 // Start with zero

	fmt.Println("Analyzing detections...")

	for i := 0; i < numAnchors; i++ {
		// Calculate indices for the i-th box in a flat array
		// Offsets: 0*8400, 1*8400, 2*8400...
		idxX := i
		idxY := i + numAnchors
		idxW := i + numAnchors*2
		idxH := i + numAnchors*3
		idxScore := i + numAnchors*4

		score := outputData[idxScore]

		// Filter: if confidence is less than 50%, skip
		if score < 0.5 {
			continue
		}

		// If we found a better candidate than the previous one - remember it
		if score > bestDetect.Conf {
			bestDetect = Detection{
				X:      outputData[idxX],
				Y:      outputData[idxY],
				Width:  outputData[idxW],
				Height: outputData[idxH],
				Conf:   score,
			}
		}
	}

	if bestDetect.Conf > 0.0 {
		fmt.Printf("\nRARE FIND FOUND!\n")
		fmt.Printf("Object: Lego Raptor\n")
		fmt.Printf("Confidence: %.2f%%\n", bestDetect.Conf*100)
		fmt.Printf("Box (640x640 scale): X=%.1f, Y=%.1f, W=%.1f, H=%.1f\n",
			bestDetect.X, bestDetect.Y, bestDetect.Width, bestDetect.Height)

		// Simple ASCII visualization of the position
		printAsciiLocation(bestDetect.X, bestDetect.Y)
	} else {
		fmt.Println("No objects detected above 50% confidence.")
	}
}

// A small utility for console visualization
func printAsciiLocation(x, y float32) {
	fmt.Println("\nPosition in Frame:")
	// 640 / 10 = 64 (grid size)
	gridX := int(x / 64)
	gridY := int(y / 64)

	for r := 0; r < 10; r++ {
		fmt.Print("[")
		for c := 0; c < 10; c++ {
			if r == gridY && c == gridX {
				fmt.Print("X") // Object is here
			} else {
				fmt.Print(" ")
			}
		}
		fmt.Println("]")
	}
}

// loadAndPreprocess reads the file, resizes to 640x640, normalizes and changes the format HWC -> CHW
func loadAndPreprocess(filename string) ([]float32, error) {
	// 1. Open the file
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	// 2. Decode (jpg/png)
	src, _, err := image.Decode(f)
	if err != nil {
		return nil, err
	}

	// 3. Resize (Scaling)
	// Create an empty 640x640 canvas
	dst := image.NewRGBA(image.Rect(0, 0, ModelInputSize, ModelInputSize))

	// Using CatmullRom for quality reduction (better than NearestNeighbor)
	draw.CatmullRom.Scale(dst, dst.Bounds(), src, src.Bounds(), draw.Over, nil)

	// 4. Convert to Tensor (CHW + Normalize 0.0-1.0)
	// We need a flat float32 array of length 3*640*640
	totalPixels := ModelInputSize * ModelInputSize
	finalData := make([]float32, ModelChannels*totalPixels)

	// We go through the pixels and fill the channels separately
	// The array structure should be: [All Red] + [All Green] + [All Blue]

	for y := 0; y < ModelInputSize; y++ {
		for x := 0; x < ModelInputSize; x++ {
			// Get the pixel color
			r, g, b, _ := dst.At(x, y).RGBA()

			// RGBA() returns values in the range 0-65535, we need 0-255, and then 0.0-1.0
			// (v >> 8) divides by 256
			rF := float32(r>>8) / 255.0
			gF := float32(g>>8) / 255.0
			bF := float32(b>>8) / 255.0

			// Calculate indices for a flat array
			// Pixel index within a single channel
			pixelIndex := y*ModelInputSize + x

			// Fill CHW (Planar) format
			finalData[pixelIndex] = rF               // 1st block: Red
			finalData[pixelIndex+totalPixels] = gF   // 2nd block: Green
			finalData[pixelIndex+totalPixels*2] = bF // 3rd block: Blue
		}
	}

	return finalData, nil
}
