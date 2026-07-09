// The Swift Programming Language
// https://docs.swift.org/swift-book

import Foundation
import MLX
import MLXNN

@main
struct nanogpt {
    static let text = loadData()
    
    static func main() {
        
        print("length of dataset in characters: \(text.count)")

        print(text.prefix(1000))
        
        let chars = Set(text).sorted()
        let vocabSize = chars.count
        print("chars: \(chars)")
        print("vocab size: \(vocabSize)")
        
        let (stoi, itos) = buildMappings(from: chars)
        
        print(encode("hii there", using: stoi))
        print(decode(encode("hii there", using: stoi), using: itos))
        
        let data = makeDataTensor(from: text, using: stoi)
        
        printTensor(data, head: 1000)
        
        let batchSize = 4
        let blockSize = 8
        MLXRandom.seed(1337)
        
        let (inputBatch, targetBatch) = getBatch(from: .training, using: stoi, batchSize: batchSize, blockSize: blockSize)
        print("inputs: ")
        print(inputBatch.shape)
        print(inputBatch.asArray(Int32.self))
        
        print("targets: ")
        print(targetBatch.shape)
        print(targetBatch.asArray(Int32.self))
        
        
        print("---")
        
        for b in 0..<batchSize {
            for t in 0..<blockSize {
                let context = inputBatch[b][0...t]
                let target = targetBatch[b][t]
                
                print("when context is \(context.asArray(Int32.self)), target is \(target.item(Int32.self))")
            }
        }
        
        let model = BigramLanguageModel(vocabSize: vocabSize)
        
        let (logits, loss) = model(inputBatch, targetBatch)
        
        print("logits: \(logits.shape)")
        print("loss: \(String(describing: loss?.item(Float.self)))")
    }
    
    private class BigramLanguageModel: Module {
        let vocabSize: Int
        let tokenEmbeddingTable: Embedding
        
        init(vocabSize: Int) {
            self.vocabSize = vocabSize
            self.tokenEmbeddingTable = Embedding(embeddingCount: vocabSize, dimensions: vocabSize)
            super.init()
        }
        
        func callAsFunction(_ inputs: MLXArray, _ targets: MLXArray? = nil) -> (MLXArray, MLXArray?) {
            let logits = tokenEmbeddingTable(inputs)
            
            if let targets = targets {
                let b = logits.shape[0]
                let t = logits.shape[1]
                let c = logits.shape[2]
                
                let logitsFlat = logits.reshaped([b * t, c])
                let targetsFlat = targets.reshaped([b * t])
                let loss = crossEntropy(logits: logitsFlat, targets: targetsFlat, reduction: .mean)
                
                return (logitsFlat, loss)
            } else {
                // generate needs the (B, T, C) shape
                return (logits, nil)
            }
        }
        
        // Generate `maxNewToken` new tokens starting from `inputs`.
        // `inputs` has shape (B, T): B sequences, each currently T tokens long.
        // Each step appends ONE new token to every sequence, so T grows by 1 each loop.
        private func generate(_ inputs: MLXArray, maxNewToken: Int) -> MLXArray {
            // The parameter is a `let`, but we need to grow it each step, so copy into a `var`.
            var inputs = inputs

            for _ in 0..<maxNewToken {
                // Run the model. With no targets it returns logits of shape (B, T, C) and a nil loss.
                // logits = a score for every candidate character, at every position, for every sequence.
                let (logits, _) = self(inputs)

                // We only append to the END, so we only need the LAST position's scores.
                // Index: B -> keep all (0...), T -> take only the last one (-1, this axis disappears), C -> keep all (0...)
                // (B, T, C) -> (B, C)
                let lastLogits = logits[0..., -1, 0...]

                // Sample one token per sequence based on those scores.
                // `categorical` applies softmax internally (scores -> probabilities) and then draws by probability,
                // so we can feed raw logits directly. Result shape: (B,) — one token id per sequence.
                let nextToken = MLXRandom.categorical(lastLogits)

                // Glue the new token onto the end of each sequence.
                // First add a trailing dimension so shapes line up: (B,) -> (B, 1)
                // Then concatenate along axis 1 (the time axis): (B, T) + (B, 1) -> (B, T + 1)
                inputs = concatenated([inputs, nextToken[0..., .newAxis]], axis: 1)
            }

            // Final shape: (B, T + maxNewToken)
            return inputs
        }
    }
    
    private static func getBatch(from dataset: Dataset, using stoi: [Character : Int], batchSize: Int, blockSize: Int) -> (MLXArray, MLXArray) {
        let data = getData(from: dataset, using: stoi)
        var inputRows = [MLXArray]()
        var targetRows = [MLXArray]()
        
        for _ in 0..<batchSize {
//            let start = Int.random(in: 0..<data.shape[0] - blockSize)
            let start = MLXRandom.randInt(0 ..< data.shape[0] - blockSize).item(Int.self)
            
            let inputs = data[start..<start + blockSize]
            let targets = data[start + 1..<start + blockSize + 1]
            
            inputRows.append(inputs)
            targetRows.append(targets)
        }
        
        return (stacked(inputRows), stacked(targetRows))
    }
    
    private static func getData(from dataset: Dataset, using stoi: [Character : Int]) -> MLXArray {
        let data = makeDataTensor(from: text, using: stoi)
        let size = data.shape[0]
        let trainSize = size * 9 / 10
        
        switch dataset {
        case .training:
            return data[0..<trainSize]
        case .validation:
            return data[trainSize..<size]
        }
    }
    
    private enum Dataset {
        case training
        case validation
    }
    
    private static func loadData() -> String {
        guard let url = Bundle.module.url(forResource: "input", withExtension: "txt"),
              let data = try? String(contentsOf: url, encoding: .utf8) else {
            return "Fail to load data"
        }
        
        return data
    }
    
    private static func buildMappings(from chars: [Character]) -> (stoi: [Character : Int], itos: [Int : Character]){
        var stoi = [Character : Int]()
        var itos = [Int : Character]()
        for (i, char) in chars.enumerated() {
            stoi[char] = i
            itos[i] = char
        }
        
        return (stoi, itos)
    }
    
    private static func encode(_ text: String, using dictionary: [Character : Int]) -> [Int] {
        return text.compactMap { dictionary[$0] }
    }
    
    private static func decode(_ code: [Int], using dictionary: [Int : Character]) -> String {
        let charArray = code.compactMap { dictionary[$0] }
        return String(charArray)
    }
    
    /// Encode text into an integer tensor (token ids) for the model to consume
    private static func makeDataTensor(from text: String, using stoi: [Character: Int]) -> MLXArray {
        MLXArray(encode(text, using: stoi))
    }
    
    private static func printTensor(_ mlxAray: MLXArray, head: Int = 10) {
        print("shape:", mlxAray.shape, " dtype:", mlxAray.dtype)
        let n = min(head, mlxAray.shape[0])
        let preview = mlxAray[0 ..< n].asArray(Int32.self)
        print("first \(n):", preview)
    }
}
