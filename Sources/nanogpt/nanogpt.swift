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
    }
    
    private static func loadData() -> String {
        guard let url = Bundle.module.url(forResource: "input", withExtension: "txt"),
              let data = try? String(contentsOf: url, encoding: .utf8) else {
            return "Fail to load data"
        }
        
        return data
    }
}
