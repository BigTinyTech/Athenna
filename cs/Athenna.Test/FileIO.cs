using System;
using Xunit;

namespace Athenna.Test
{
    public class FileIO
    {
        [Fact]
        public void SaveFile()
        {
            var nn = new Athenna(new int[] { 1, 1 }, new Activations[] { Activations.TanH });
            nn.Save("c:/Data/test.athenna");
        }
        [Fact]
        public void LoadFile()
        {
            var nn = Athenna.Load("c:/Data/test.athenna");
        }
    }
}
