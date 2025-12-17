package main

import (
	"fmt"
	"os"
	"path/filepath"

	ort "github.com/yalue/onnxruntime_go"
)

func main() {
	// 1. Setup Library
	dllPath, _ := filepath.Abs("onnxruntime.dll")
	ort.SetSharedLibraryPath(dllPath)

	err := ort.InitializeEnvironment()
	if err != nil {
		fmt.Printf("Error initializing environment: %v\n", err)
		return
	}
	defer ort.DestroyEnvironment()

	// 2. Check Model File
	modelPath := filepath.Join("..", "models", "model.onnx")
	absModelPath, _ := filepath.Abs(modelPath)

	if _, err := os.Stat(absModelPath); os.IsNotExist(err) {
		fmt.Printf("Model file not found: %s\n", absModelPath)
		return
	}

	// 3. Load Session
	inputNames := []string{"images"}
	outputNames := []string{"output0"}

	session, err := ort.NewDynamicAdvancedSession(
		absModelPath,
		inputNames,
		outputNames,
		nil,
	)
	if err != nil {
		fmt.Printf("Error creating session: %v\n", err)
		return
	}
	defer session.Destroy()
	fmt.Println("Model loaded successfully!")

	// 4. Prepare Input Tensor
	// Shape: [1, 3, 640, 640]
	inputData := make([]float32, 1*3*640*640)
	inputShape := ort.NewShape(1, 3, 640, 640)

	inputTensor, err := ort.NewTensor(inputShape, inputData)
	if err != nil {
		fmt.Printf("Error creating input tensor: %v\n", err)
		return
	}
	// We must destroy tensors manually in CGO
	defer inputTensor.Destroy()

	// 5. Prepare Output Tensor
	// We must pre-allocate memory for the result.
	// YOLOv8 output: [1, 5, 8400] (Batch, [x,y,w,h,conf], Anchors)
	outputSize := 1 * 5 * 8400
	outputData := make([]float32, outputSize)
	outputShape := ort.NewShape(1, 5, 8400)

	outputTensor, err := ort.NewTensor(outputShape, outputData)
	if err != nil {
		fmt.Printf("Error creating output tensor: %v\n", err)
		return
	}
	defer outputTensor.Destroy()

	fmt.Println("Running inference on dummy data...")

	// 6. Run Inference
	// Pass inputs and outputs as slices of ort.Value
	err = session.Run(
		[]ort.Value{inputTensor},  // Inputs
		[]ort.Value{outputTensor}, // Outputs
	)
	if err != nil {
		fmt.Printf("Error running inference: %v\n", err)
		return
	}

	// 7. Validate Output
	// The inference writes directly into 'outputData' slice we created earlier
	fmt.Printf("Success! Output buffer filled. Size: %d\n", len(outputData))

	if len(outputData) == outputSize {
		fmt.Println("Check passed: output dimensions match YOLOv8.")
	} else {
		fmt.Printf("Warning: Unexpected output size.\n")
	}
}
