using System.Text.RegularExpressions;
using iText.Kernel.Pdf;
using iText.Kernel.Pdf.Canvas.Parser;
using iText.Kernel.Pdf.Canvas.Parser.Listener;

Console.WriteLine("=== PDF Document Similarity Analyzer ===\n");

// Get all PDF files from Assets folder
var assetsPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Assets");
var pdfFiles = Directory.GetFiles(assetsPath, "*.pdf");

if (pdfFiles.Length < 2)
{
    Console.WriteLine("Need at least 2 PDF files in the Assets folder for comparison.");
    return;
}

Console.WriteLine($"Found {pdfFiles.Length} PDF files:\n");
foreach (var file in pdfFiles)
{
    Console.WriteLine($"  - {Path.GetFileName(file)}");
}
Console.WriteLine();

// Extract text from all PDFs
var documents = new Dictionary<string, string>();
foreach (var pdfFile in pdfFiles)
{
    var fileName = Path.GetFileName(pdfFile);
    var text = DocumentTextExtractor.ExtractText(pdfFile);
    documents[fileName] = text;
    Console.WriteLine($"Extracted {text.Length} characters from first page of {fileName}");
}
Console.WriteLine();

// Compare each document with every other document
Console.WriteLine("=== Document Similarity Comparison ===\n");
var fileNames = documents.Keys.ToArray();

for (int i = 0; i < fileNames.Length; i++)
{
    for (int j = i + 1; j < fileNames.Length; j++)
    {
        var doc1 = fileNames[i];
        var doc2 = fileNames[j];
        var text1 = documents[doc1];
        var text2 = documents[doc2];

        // Console.WriteLine($"Comparing: '{doc1}' vs '{doc2}'");
        // Console.WriteLine(new string('-', 60));
        //
        // var cosineSim = TextSimilarity.CosineSimilarity(text1, text2);
        // Console.WriteLine($"  Cosine Similarity:        {cosineSim:P2}");
        //
        // var levenshtein = TextSimilarity.LevenshteinDistance(text1, text2);
        // Console.WriteLine($"  Levenshtein Distance:     {levenshtein:N0} edits");
        //
        // var normalizedLev = TextSimilarity.NormalizedLevenshtein(text1, text2);
        // Console.WriteLine($"  Normalized Levenshtein:   {normalizedLev:P2}");
        //
        // var jaccard = TextSimilarity.JaccardSimilarity(text1, text2);
        // Console.WriteLine($"  Jaccard Similarity:       {jaccard:P2}");

        var embeddingSim = TextSimilarity.EmbeddingSimilarity(text1, text2, documents.Values.ToList());
        Console.WriteLine($"  Embedding Similarity:     {embeddingSim:P2}");

        Console.WriteLine();
    }
}

public static class DocumentTextExtractor
{
    public static string ExtractText(string pdfPath)
    {
        try
        {
            using var pdfReader = new PdfReader(pdfPath);
            using var pdfDocument = new PdfDocument(pdfReader);
            
            // Extract only the first page
            if (pdfDocument.GetNumberOfPages() > 0)
            {
                var page = pdfDocument.GetPage(1);
                var strategy = new SimpleTextExtractionStrategy();
                var pageText = PdfTextExtractor.GetTextFromPage(page, strategy);
                return pageText;
            }
            
            return string.Empty;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error extracting text from {Path.GetFileName(pdfPath)}: {ex.Message}");
            return string.Empty;
        }
    }
}

public static class TextSimilarity
{
    /// <summary>
    /// Calculates cosine similarity between two texts based on word frequency vectors.
    /// Returns a value between 0 (completely different) and 1 (identical).
    /// </summary>
    public static double CosineSimilarity(string text1, string text2)
    {
        var words1 = GetWords(text1);
        var words2 = GetWords(text2);

        var allWords = words1.Keys.Union(words2.Keys).ToList();

        double dotProduct = 0;
        double magnitude1 = 0;
        double magnitude2 = 0;

        foreach (var word in allWords)
        {
            var freq1 = words1.GetValueOrDefault(word, 0);
            var freq2 = words2.GetValueOrDefault(word, 0);

            dotProduct += freq1 * freq2;
            magnitude1 += freq1 * freq1;
            magnitude2 += freq2 * freq2;
        }

        if (magnitude1 == 0 || magnitude2 == 0)
            return 0;

        return dotProduct / (Math.Sqrt(magnitude1) * Math.Sqrt(magnitude2));
    }

    /// <summary>
    /// Calculates Levenshtein distance (minimum number of single-character edits).
    /// Lower values indicate more similarity.
    /// </summary>
    public static int LevenshteinDistance(string text1, string text2)
    {
        if (string.IsNullOrEmpty(text1))
            return text2?.Length ?? 0;

        if (string.IsNullOrEmpty(text2))
            return text1.Length;

        var matrix = new int[text1.Length + 1, text2.Length + 1];

        for (int i = 0; i <= text1.Length; i++)
            matrix[i, 0] = i;

        for (int j = 0; j <= text2.Length; j++)
            matrix[0, j] = j;

        for (int i = 1; i <= text1.Length; i++)
        {
            for (int j = 1; j <= text2.Length; j++)
            {
                int cost = text1[i - 1] == text2[j - 1] ? 0 : 1;

                matrix[i, j] = Math.Min(
                    Math.Min(matrix[i - 1, j] + 1, matrix[i, j - 1] + 1),
                    matrix[i - 1, j - 1] + cost
                );
            }
        }

        return matrix[text1.Length, text2.Length];
    }

    /// <summary>
    /// Normalized Levenshtein similarity (0-1, higher is more similar).
    /// </summary>
    public static double NormalizedLevenshtein(string text1, string text2)
    {
        var maxLength = Math.Max(text1.Length, text2.Length);
        if (maxLength == 0)
            return 1.0;

        var distance = LevenshteinDistance(text1, text2);
        return 1.0 - (double)distance / maxLength;
    }

    /// <summary>
    /// Calculates Jaccard similarity based on word sets.
    /// Returns a value between 0 (no common words) and 1 (identical word sets).
    /// </summary>
    public static double JaccardSimilarity(string text1, string text2)
    {
        var words1 = GetWords(text1).Keys.ToHashSet();
        var words2 = GetWords(text2).Keys.ToHashSet();

        if (words1.Count == 0 && words2.Count == 0)
            return 1.0;

        var intersection = words1.Intersect(words2).Count();
        var union = words1.Union(words2).Count();

        return union == 0 ? 0 : (double)intersection / union;
    }

    /// <summary>
    /// Calculates similarity using TF-IDF embeddings.
    /// Returns a value between 0 (completely different) and 1 (identical).
    /// </summary>
    public static double EmbeddingSimilarity(string text1, string text2, List<string> corpus)
    {
        var words1 = GetWords(text1);
        var words2 = GetWords(text2);
        
        // Calculate TF-IDF vectors
        var tfidf1 = CalculateTfIdf(words1, corpus);
        var tfidf2 = CalculateTfIdf(words2, corpus);
        
        // Calculate cosine similarity between TF-IDF vectors
        var allTerms = tfidf1.Keys.Union(tfidf2.Keys).ToList();
        
        double dotProduct = 0;
        double magnitude1 = 0;
        double magnitude2 = 0;
        
        foreach (var term in allTerms)
        {
            var value1 = tfidf1.GetValueOrDefault(term, 0.0);
            var value2 = tfidf2.GetValueOrDefault(term, 0.0);
            
            dotProduct += value1 * value2;
            magnitude1 += value1 * value1;
            magnitude2 += value2 * value2;
        }
        
        if (magnitude1 == 0 || magnitude2 == 0)
            return 0;
        
        return dotProduct / (Math.Sqrt(magnitude1) * Math.Sqrt(magnitude2));
    }
    
    private static Dictionary<string, double> CalculateTfIdf(Dictionary<string, int> wordCounts, List<string> corpus)
    {
        var tfidf = new Dictionary<string, double>();
        var totalWords = wordCounts.Values.Sum();
        
        foreach (var (word, count) in wordCounts)
        {
            // Term Frequency (TF)
            var tf = (double)count / totalWords;
            
            // Inverse Document Frequency (IDF)
            var documentsContainingWord = corpus.Count(doc => 
                Regex.IsMatch(doc.ToLowerInvariant(), $@"\b{Regex.Escape(word)}\b"));
            
            var idf = Math.Log(1 + (double)corpus.Count / (1 + documentsContainingWord));
            
            // TF-IDF
            tfidf[word] = tf * idf;
        }
        
        return tfidf;
    }

    private static Dictionary<string, int> GetWords(string text)
    {
        var words = Regex.Split(text.ToLowerInvariant(), @"\W+")    
            .Where(w => !string.IsNullOrWhiteSpace(w) && !IsStopWord(w));

        var wordCounts = new Dictionary<string, int>();
        foreach (var word in words)
        {
            wordCounts[word] = wordCounts.GetValueOrDefault(word, 0) + 1;
        }

        return wordCounts;
    }
    
    private static bool IsStopWord(string word)
    {
        // Common English stopwords
        var stopWords = new HashSet<string>
        {
            "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", 
            "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", 
            "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", 
            "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", 
            "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", 
            "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", 
            "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", 
            "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", 
            "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", 
            "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", 
            "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", 
            "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", 
            "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", 
            "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", 
            "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", 
            "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", 
            "yours", "yourself", "yourselves"
        };
        
        return stopWords.Contains(word);
    }
}
