param(
    [Parameter(Mandatory=$true)]
    [string]$WavPath,
    [Parameter(Mandatory=$true)]
    [string]$JsonPath,
    [int]$MaxSeconds = 0
)

Add-Type -AssemblyName System.Speech

if (-not (Test-Path $WavPath)) {
    Write-Error "Wav file not found: $WavPath"
    exit 1
}

$fullWavPath = (Resolve-Path $WavPath).Path
$stream = [System.IO.File]::OpenRead($fullWavPath)
$engine = New-Object System.Speech.Recognition.SpeechRecognitionEngine
$grammar = New-Object System.Speech.Recognition.DictationGrammar
$engine.LoadGrammar($grammar)
$engine.SetInputToWaveStream($stream)

$headerSize = 44
$bytesPerSec = 32000 # 16kHz, 16-bit mono

$words = [System.Collections.Generic.List[PSCustomObject]]::new()
$wordId = 1

try {
    while ($true) {
        $posStart = [Math]::Max(0.0, ($stream.Position - $headerSize) / $bytesPerSec)
        if ($MaxSeconds -gt 0 -and $posStart -ge $MaxSeconds) {
            break
        }
        $res = $null
        try {
            $res = $engine.Recognize()
        } catch {
            break
        }
        if ($null -eq $res) {
            if ($stream.Position -ge ($stream.Length - 1000)) {
                break
            }
            $stream.Position = [Math]::Min($stream.Length, $stream.Position + 16000)
            continue
        }
        $posEnd = [Math]::Max($posStart + 0.1, ($stream.Position - $headerSize) / $bytesPerSec)
        $phraseDur = $posEnd - $posStart
        $count = $res.Words.Count

        if ($count -gt 0) {
            $wDur = $phraseDur / $count
            for ($i = 0; $i -lt $count; $i++) {
                $w = $res.Words[$i]
                $wStart = [Math]::Round($posStart + ($i * $wDur), 3)
                $wEnd = [Math]::Round($posStart + (($i + 1) * $wDur), 3)
                $wordObj = [PSCustomObject]@{
                    id = ("w_" + $wordId.ToString("D4"))
                    word = $w.Text
                    start = $wStart
                    end = $wEnd
                    confidence = [Math]::Round($w.Confidence, 3)
                }
                $words.Add($wordObj)
                $wordId++
            }
        }
    }
} finally {
    $engine.Dispose()
    $stream.Dispose()
}

$json = ConvertTo-Json -InputObject @($words) -Depth 3
if ($words.Count -eq 1 -and -not $json.Trim().StartsWith("[")) {
    $json = "[$json]"
}
$json | Set-Content -Path $JsonPath -Encoding UTF8
Write-Host "Transcribed $($words.Count) words to $JsonPath"
