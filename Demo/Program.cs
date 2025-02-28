#pragma warning disable SKEXP0001
#pragma warning disable SKEXP0050
#pragma warning disable SKEXP0070

using System.Collections;

using Demo;
using Demo.Extensions;
using Demo.TextSplitters;

using Microsoft.Extensions.AI;
using Microsoft.Extensions.VectorData;
using Microsoft.ML.Tokenizers;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Embeddings;
using Microsoft.SemanticKernel.Text;

using Tabula.Detectors;
using Tabula.Extractors;
using Tabula;
using UglyToad.PdfPig;
using System.Text;
using System;
using Microsoft.SemanticKernel.Plugins.Memory;
using Microsoft.SemanticKernel.Memory;


var textSplitter = new RecursiveCharacterTextSplitter();

const string CollectionName = "Documents";

var ollamaEndpoint = new Uri(@"http://localhost:11434");

const string modelInferenceId = @"phi4:latest";
const string modelEmbeddingId = @"nomic-embed-text:latest";

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

Console.WriteLine("Bienvenidos a la Global AI de Madrid 2025");

string resourcesPath = Path.Combine(Directory.GetCurrentDirectory(), @"Resources");

// Get all files from the Resources folder
string[] files = Directory.GetFiles(resourcesPath);

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
        ConsoleColor originalColor = Console.ForegroundColor;
        Console.ForegroundColor = ConsoleColor.Yellow;
        Console.WriteLine($"Warning: Unsupported file format: {Path.GetFileName(file)}");
        Console.ForegroundColor = originalColor;
    }
}

const string SystemPrompt = @"""
You are a helpful assistant.
Provide accurate, well-reasoned responses based on the given context.
If the context is insufficient to answer the question, say so.
Always respond in the same language as the question.

Context: {{$context}}

Question: {{$input}}

""";

while (true)
{
    Console.Write(@"Question: ");
    var userQuestion = Console.ReadLine();

    if (string.IsNullOrEmpty(userQuestion))
    {
        continue;
    }

    var search = collection as IVectorizableTextSearch<Document>;
    var searchResult = await search!.VectorizableTextSearchAsync(userQuestion, new() { Top = 1 });
    var resultRecords = await searchResult.Results.ToListAsync();

    var arguments = new KernelArguments()
    {
        { @"input", userQuestion },
        { @"context", string.Join("\n\n", resultRecords.Select(rr => rr.Record.Text)) },
    };

    var response = kernel.InvokePromptStreamingAsync(SystemPrompt, arguments);

    Console.WriteLine(@"Answering...");
    await foreach (var item in response)
    {
        Console.Write(item);
    }

    Console.WriteLine($"Ref: {resultRecords[0].Record.Id}\n");

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
