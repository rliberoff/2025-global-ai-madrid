#pragma warning disable SKEXP0001

using Microsoft.Extensions.VectorData;
using Microsoft.SemanticKernel.Embeddings;

namespace Demo;

public static class TextEmbeddingVectorStoreExtensions
{
    public static IVectorStore UseTextEmbeddingGeneration(this IVectorStore vectorStore, ITextEmbeddingGenerationService textEmbeddingGenerationService)
    {
        return new TextEmbeddingVectorStore(vectorStore, textEmbeddingGenerationService);
    }
}
