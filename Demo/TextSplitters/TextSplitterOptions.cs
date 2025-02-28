namespace Demo.TextSplitters;

public class TextSplitterOptions
{
    public int ChunkOverlap { get; init; } = 50;

    public int ChunkSize { get; init; } = 700;

    public IList<string> Separators { get; init; }
}
