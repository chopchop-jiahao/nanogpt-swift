// The Swift Programming Language
// https://docs.swift.org/swift-book

import Foundation

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
        
        let (stoi, itos) = buildDictionaries(from: chars)
        
        print(encode("hii there", using: stoi))
        print(decode(encode("hii there", using: stoi), using: itos))
    }
    
    private static func loadData() -> String {
        guard let url = Bundle.module.url(forResource: "input", withExtension: "txt"),
              let data = try? String(contentsOf: url, encoding: .utf8) else {
            return "Fail to load data"
        }
        
        return data
    }
    
    private static func buildDictionaries(from chars: [Character]) -> (stoi: [Character : Int], itos: [Int : Character]){
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
}
