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
        print("loss: \(loss.item(Float.self))")
    }
    
    private class BigramLanguageModel: Module {
        let vocabSize: Int
        let tokenEmbeddingTable: Embedding
        
        init(vocabSize: Int) {
            self.vocabSize = vocabSize
            self.tokenEmbeddingTable = Embedding(embeddingCount: vocabSize, dimensions: vocabSize)
            super.init()
        }
        
        func callAsFunction(_ inputs: MLXArray, _ targets: MLXArray) -> (MLXArray, MLXArray) {
            let logits = tokenEmbeddingTable(inputs)
            let b = logits.shape[0]
            let t = logits.shape[1]
            let c = logits.shape[2]
            
            let logitsFlat = logits.reshaped([b * t, c])
            let targetsFlat = targets.reshaped([b * t])
            
            let loss = crossEntropy(logits: logitsFlat, targets: targetsFlat, reduction: .mean)
            
            return (logitsFlat, loss)
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
