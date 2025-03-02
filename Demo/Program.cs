#pragma warning disable SKEXP0001
#pragma warning disable SKEXP0050
#pragma warning disable SKEXP0070

using Demo;

using Microsoft.Extensions.VectorData;
using Microsoft.ML.Tokenizers;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Embeddings;

using UglyToad.PdfPig;

const string CollectionName = "Documents";

const string modelInferenceId = @"phi4:latest";
//const string modelInferenceId = @"phi-rate-4:4.0";
const string modelEmbeddingId = @"nomic-embed-text:latest";

string[] files = Directory.GetFiles(Path.Combine(Directory.GetCurrentDirectory(), @"Resources"));

var ollamaEndpoint = new Uri(@"http://localhost:11434");

var tokenizer = TiktokenTokenizer.CreateForEncoding(@"cl100k_base");

Kernel kernel = Kernel.CreateBuilder()
                      .AddOllamaTextEmbeddingGeneration(modelEmbeddingId, ollamaEndpoint)
                      .AddOllamaChatCompletion(modelInferenceId, ollamaEndpoint)    
                      .AddInMemoryVectorStore()
                      .Build();

var textEmbeddingGenerationService = kernel.GetRequiredService<ITextEmbeddingGenerationService>();

var vectorStore = kernel.GetRequiredService<IVectorStore>().UseTextEmbeddingGeneration(textEmbeddingGenerationService);

var collection = vectorStore.GetCollection<string, Document>(CollectionName);
await collection.CreateCollectionIfNotExistsAsync();

ConsoleColor originalColor = Console.ForegroundColor;

Console.ForegroundColor = ConsoleColor.Cyan;
Console.WriteLine("Bienvenidos a la Global AI de Madrid 2025\n");
Console.ForegroundColor = ConsoleColor.Magenta;
Console.WriteLine("Cargando documentos en base de datos vectorial local. Por favor espere...\n");
Console.ForegroundColor = originalColor;

// Process each file
foreach (string file in files)
{
    string extension = Path.GetExtension(file).ToLower();

    if (extension == @".txt")
    {
        var content = await File.ReadAllTextAsync(file);

        await collection.UpsertAsync(new Document()
        {
            Id = Path.GetFileName(file),
            Name = Path.GetFileName(file),
            Text = content,
        });
    }
    else if (extension == @".pdf")
    {
        using var document = PdfDocument.Open(file, new ParsingOptions() { ClipPaths = true });

        var items = document.GetPages().Where(p => p.GetWords().Count() > 10)
                                      .Select(page => new Document()
                                      {
                                          Id = $@"{page.Number}-{Path.GetFileName(file)}",
                                          Name = Path.GetFileName(file),
                                          Text = string.Join(" ", page.GetWords().Select(w => w.Text)),
                                      })
                                      .ToList();

        await collection.UpsertBatchAsync(items).ToListAsync();
    }
    else
    {
        Console.ForegroundColor = ConsoleColor.Yellow;
        Console.WriteLine($"Warning: Unsupported file format: {Path.GetFileName(file)}");
        Console.ForegroundColor = originalColor;
    }
}

const string MetaPrompt = @"""
Provide accurate responses based only on the given context.
Do not answer from your own knowledge base.
If the context is insufficient to answer the question, say so.
Always respond in the same language as the question.

Context:

";

var chatService = kernel.GetRequiredService<IChatCompletionService>();
var search = collection as IVectorizableTextSearch<Document>;

while (true)
{
    Console.ForegroundColor = ConsoleColor.Yellow;
    Console.Write(@"Question: ");
    var userQuestion = Console.ReadLine();

    if (string.IsNullOrEmpty(userQuestion))
    {
        continue;
    }
    
    var searchResult = await search!.VectorizableTextSearchAsync(userQuestion, new() { Top = 1 });
    var resultRecords = await searchResult.Results.ToListAsync();

    var chatHistory = new ChatHistory();
    chatHistory.AddDeveloperMessage(MetaPrompt + string.Join("\n\n", resultRecords.Select(rr => rr.Record.Text)));
    chatHistory.AddUserMessage(userQuestion);

    var response = chatService.GetStreamingChatMessageContentsAsync(chatHistory);

    Console.ForegroundColor = ConsoleColor.Green;
    Console.WriteLine(@"Answering...");
    await foreach (var item in response)
    {
        Console.Write(item.Content);
    }

    Console.ForegroundColor = ConsoleColor.Red;
    Console.WriteLine($"\n\nRef: {resultRecords[0].Record.Id}");

    Console.ForegroundColor = originalColor;
    Console.WriteLine(string.Empty);
    Console.WriteLine(string.Empty);
}

[AttributeUsage(AttributeTargets.Property, AllowMultiple = false, Inherited = true)]
public sealed class GenerateTextEmbeddingAttribute : Attribute
{
    public GenerateTextEmbeddingAttribute(string sourcePropertyName)
    {
        this.SourcePropertyNames = [sourcePropertyName];
    }

    public GenerateTextEmbeddingAttribute(string[] sourcePropertyNames)
    {
        this.SourcePropertyNames = sourcePropertyNames;
    }

    public string[] SourcePropertyNames { get; }
}

public class Document
{
    [VectorStoreRecordKey]
    public string Id { get; set; }

    [VectorStoreRecordData(IsFilterable = true)]
    public string Name { get; set; }

    [VectorStoreRecordData(IsFullTextSearchable = true)]
    public string Text { get; set; }

    [GenerateTextEmbedding(nameof(Text))]
    [VectorStoreRecordVector(768, DistanceFunction.CosineSimilarity, IndexKind.Dynamic)]
    public ReadOnlyMemory<float>? Embedding { get; set; }
}
