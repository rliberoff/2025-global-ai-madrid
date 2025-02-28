using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Demo.TextSplitters;

public class RecursiveCharacterTextSplitter
{
    public static readonly string[] DefaultSeparators = { ".", "!", "?", ";", ":", ")", "\r\n", "\n" };

    public RecursiveCharacterTextSplitter() : this(new TextSplitterOptions())
    {
    }

    public RecursiveCharacterTextSplitter(TextSplitterOptions options)
    {
        ChunkOverlap = options.ChunkOverlap;
        ChunkSize = options.ChunkSize;
        Separators = options.Separators?.Any() == true ? options.Separators : new List<string>(DefaultSeparators);

        if (ChunkOverlap > ChunkSize)
        {
            throw new InvalidOperationException(@"Configured value for chunk overlap is greater than configured value for chunk size. It must be smaller!");
        }
    }

    /// <inheritdoc/>
    public int ChunkOverlap { get; }

    /// <inheritdoc/>
    public int ChunkSize { get; }

    /// <inheritdoc/>
    public IList<string> Separators { get; }

    /// <inheritdoc/>
    public IEnumerable<string> Split(string text, Func<string, int> lengthFunction)
    {
        return Split(text, lengthFunction, new TextSplitterOptions()
        {
            ChunkOverlap = ChunkOverlap,
            ChunkSize = ChunkSize,
            Separators = Separators,
        });
    }

    public IEnumerable<string> Split(string text, Func<string, int> lengthFunction, TextSplitterOptions options)
    {
        var chunks = new List<string>();

        if (string.IsNullOrWhiteSpace(text))
        {
            return chunks;
        }

        string? separator = null;

        foreach (var s in options.Separators)
        {
            if (s == string.Empty || text.Contains(s, StringComparison.OrdinalIgnoreCase))
            {
                separator = s;
                break;
            }
        }

        var splits = (separator != null ? text.Split(separator, StringSplitOptions.RemoveEmptyEntries) : [text]).Select(s => s.Trim());

        var goodSplits = new List<string>();

        foreach (var split in splits)
        {
            if (lengthFunction(split) < options.ChunkSize)
            {
                goodSplits.Add(split);
            }
            else
            {
                if (goodSplits.Any())
                {
                    chunks.AddRange(MergeSplits(goodSplits, separator, lengthFunction, options));
                    goodSplits = new List<string>();
                }

                var otherChunks = Split(split, lengthFunction, options);
                chunks.AddRange(otherChunks);
            }
        }

        if (goodSplits.Any())
        {
            chunks.AddRange(MergeSplits(goodSplits, separator, lengthFunction, options));
        }

        return chunks.Where(chunk => !string.IsNullOrWhiteSpace(chunk));
    }

    public string JoinChunks(IEnumerable<string> chunks, string separator)
    {
        var text = string.Join(separator, chunks).Trim();

        return text;
    }

    public IEnumerable<string> MergeSplits(IEnumerable<string> splits, string separator, Func<string, int> lengthFunction)
    {
        return MergeSplits(splits, separator, lengthFunction, new TextSplitterOptions()
        {
            ChunkOverlap = ChunkOverlap,
            ChunkSize = ChunkSize,
            Separators = Separators,
        });
    }

    public IEnumerable<string> MergeSplits(IEnumerable<string> splits, string separator, Func<string, int> lengthFunction, TextSplitterOptions options)
    {
        string chunk;
        var chunks = new List<string>();
        var currentChunks = new Queue<string>();

        var total = 0;
        var separatorLength = lengthFunction(separator);

        foreach (var split in splits)
        {
            var splitLength = lengthFunction(split);
            var hasCurrentChunks = currentChunks.Any();

            if (hasCurrentChunks && total + splitLength + separatorLength > options.ChunkSize)
            {
                chunk = JoinChunks(currentChunks, separator);

                if (chunk != null)
                {
                    chunks.Add(chunk);
                }

                // Keep on dequeuing if:
                // - There are still chunks and their length is long
                // - There is a larger chunk than the chunk overlap
                while (
                    hasCurrentChunks
                    && (total > options.ChunkOverlap || (total + splitLength + separatorLength > options.ChunkSize && total > 0)))
                {
                    total -= lengthFunction(currentChunks.Dequeue()) + (currentChunks.Count > 1 ? separatorLength : 0);
                    hasCurrentChunks = currentChunks.Any();
                }
            }

            currentChunks.Enqueue(split);
            total += splitLength + (currentChunks.Count > 1 ? separatorLength : 0);
        }

        chunk = JoinChunks(currentChunks, separator);

        if (chunk != null)
        {
            chunks.Add(chunk);
        }

        return chunks;
    }
}
