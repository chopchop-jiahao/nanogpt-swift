// The Swift Programming Language
// https://docs.swift.org/swift-book

import Foundation
import MLX

@main
struct nanogpt {
    static func main() {
        let text = loadData()
        
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
