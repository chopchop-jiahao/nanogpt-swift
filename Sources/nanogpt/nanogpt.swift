// The Swift Programming Language
// https://docs.swift.org/swift-book

import Foundation

@main
struct nanogpt {
    static func main() {
        let text = loadData()
        
        print(text)
    }
    
    private static func loadData() -> String {
        guard let url = Bundle.module.url(forResource: "input", withExtension: "txt"),
              let data = try? String(contentsOf: url, encoding: .utf8) else {
            return "Fail to load data"
        }
        
        return data
    }
}
