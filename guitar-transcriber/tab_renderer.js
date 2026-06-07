// Guitar Tab Canvas Renderer
// Renders tab data (bars/notes) onto Canvas elements with chord detection
var _tabTuningNames = ['e','B','G','D','A','E'];

// Chord pattern dictionary: interval set -> chord name
var _CHORD_PATTERNS = [
  {name:'maj',  ints:[0,4,7]},
  {name:'m',    ints:[0,3,7]},
  {name:'dim',  ints:[0,3,6]},
  {name:'aug',  ints:[0,4,8]},
  {name:'sus2', ints:[0,2,7]},
  {name:'sus4', ints:[0,5,7]},
  {name:'7',    ints:[0,4,7,10]},
  {name:'maj7', ints:[0,4,7,11]},
  {name:'m7',   ints:[0,3,7,10]},
  {name:'dim7', ints:[0,3,6,9]},
  {name:'m7b5', ints:[0,3,6,10]},
  {name:'aug7', ints:[0,4,8,10]},
  {name:'6',    ints:[0,4,7,9]},
  {name:'m6',   ints:[0,3,7,9]},
  {name:'9',    ints:[0,4,7,10,2]},
  {name:'m9',   ints:[0,3,7,10,2]},
  {name:'add9', ints:[0,4,7,2]},
  {name:'7sus4',ints:[0,5,7,10]},
];

var _NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'];

function _notesToChord(notes, capo){
  // notes: array of {string, fret, pitch} (or just pitch values)
  // capo: number of frets (0 = no capo)
  // Returns {name: 'Am', score: 0.8} or null

  if(!notes || notes.length < 2) return null;

  // Collect unique pitch classes
  var pcs = {};
  for(var i = 0; i < notes.length; i++){
    var pitch = typeof notes[i] === 'number' ? notes[i] : notes[i].pitch;
    if(pitch != null) pcs[pitch % 12] = true;
  }
  var pcList = Object.keys(pcs).map(Number);
  if(pcList.length < 2) return null;

  var best = null;
  for(var r = 0; r < 12; r++){
    // Transpose all pitch classes relative to root r
    var rel = {};
    for(var k in pcs) rel[(parseInt(k) - r + 12) % 12] = true;
    var relArr = Object.keys(rel).map(Number);

    // Try each chord pattern
    for(var ci = 0; ci < _CHORD_PATTERNS.length; ci++){
      var pat = _CHORD_PATTERNS[ci];
      var match = 0;
      for(var pi = 0; pi < pat.ints.length; pi++){
        if(rel[pat.ints[pi]]) match++;
      }
      var extra = relArr.length - pat.ints.length;
      var score = match / Math.max(pat.ints.length, relArr.length);
      // Penalty for extra notes not in chord
      if(extra > 0) score -= extra * 0.1;
      if(score > 0.5 && (!best || score > best.score)){
        best = { root: r, type: pat.name, score: score };
      }
    }
  }

  if(!best) return null;

  // Apply capo: shift root down by capo frets
  var displayRoot = (best.root - (capo || 0) + 12) % 12;
  var rootName = _NOTE_NAMES[displayRoot];
  var suffix = best.type === 'maj' ? '' : best.type;
  return rootName + suffix;
}

function detectChords(bars, capo){
  // Detect chord for each bar based on its notes
  capo = capo || 0;
  var chords = [];
  for(var i = 0; i < bars.length; i++){
    var notes = bars[i].notes || [];
    var chord = _notesToChord(notes, capo);
    chords.push({ barIndex: i, chord: chord ? chord : null });
  }

  // Smooth: remove single-bar anomalies (A ? B ? A becomes A ? A ? A)
  for(var i = 1; i < chords.length - 1; i++){
    var prev = chords[i-1].chord;
    var cur = chords[i].chord;
    var next = chords[i+1].chord;
    if(cur && prev && next && cur !== prev && prev === next){
      chords[i].chord = prev;
    }
  }

  // Fill gaps: if a bar has no chord but neighbors share the same chord
  for(var i = 1; i < chords.length - 1; i++){
    if(!chords[i].chord && chords[i-1].chord && chords[i+1].chord){
      if(chords[i-1].chord === chords[i+1].chord){
        chords[i].chord = chords[i-1].chord;
      }
    }
  }

  // Propagate: fill leading nulls with first detected chord
  var firstChord = null;
  for(var i = 0; i < chords.length; i++){
    if(chords[i].chord){ firstChord = chords[i].chord; break; }
  }
  if(firstChord){
    for(var i = 0; i < chords.length; i++){
      if(!chords[i].chord) chords[i].chord = firstChord;
      else break;
    }
  }

  return chords;
}

function renderTab(data, capo){
  var bars = data.bars || [];
  var chords = detectChords(bars, capo || 0);
  var wrap = document.getElementById('tscAtRender');
  wrap.innerHTML = '';

  if(!bars.length){
    wrap.innerHTML = '<p style="color:#6a5a80;text-align:center;padding:40px">未检测到音符</p>';
    return;
  }

  var barWidth = 230;
  var barGap = 12;
  var leftPad = 34;
  var rightPad = 10;
  var stringSpacing = 20;
  var topPad = 36;  // increased for chord label
  var bottomPad = 14;
  var barHeight = topPad + (5 * stringSpacing) + bottomPad;
  var maxBeat = 4;

  var wrapperWidth = wrap.clientWidth || 800;
  var colsPerRow = Math.max(1, Math.floor((wrapperWidth - 40) / (barWidth + barGap)));

  var rows = [];
  for(var i = 0; i < bars.length; i += colsPerRow){
    rows.push({idx: i, bars: bars.slice(i, i + colsPerRow)});
  }

  rows.forEach(function(row){
    var rowDiv = document.createElement('div');
    rowDiv.style.cssText = 'display:flex;gap:'+barGap+'px;justify-content:center;margin-bottom:20px;flex-wrap:wrap';

    row.bars.forEach(function(bar, bi){
      var canvas = document.createElement('canvas');
      var dpr = window.devicePixelRatio || 1;
      canvas.width = barWidth * dpr;
      canvas.height = barHeight * dpr;
      canvas.style.width = barWidth + 'px';
      canvas.style.height = barHeight + 'px';
      canvas.setAttribute('data-bar-idx', row.idx + bi);
      canvas.style.cssText += 'background:rgba(255,255,255,0.02);border-radius:8px;border:1px solid rgba(255,255,255,0.04)';

      var ctx = canvas.getContext('2d');
      ctx.scale(dpr, dpr);

      // Chord label above bar
      var barIdx = row.idx + bi;
      var chordData = chords[barIdx];
      if(chordData && chordData.chord){
        ctx.fillStyle = '#f48fb1';
        ctx.font = 'bold 13px monospace';
        ctx.textAlign = 'center';
        ctx.fillText(chordData.chord, barWidth / 2, topPad - 10);
      }

      // Draw 6 strings
      for(var s = 0; s < 6; s++){
        var y = topPad + s * stringSpacing;
        ctx.beginPath();
        ctx.moveTo(leftPad, y);
        ctx.lineTo(barWidth - rightPad, y);
        var isBass = s >= 3;
        ctx.strokeStyle = isBass ? 'rgba(255,255,255,0.28)' : 'rgba(255,255,255,0.14)';
        ctx.lineWidth = isBass ? 1.5 : 1;
        ctx.stroke();
      }

      // String names
      ctx.fillStyle = '#5a4a70';
      ctx.font = 'bold 10px monospace';
      ctx.textAlign = 'right';
      for(var s = 0; s < 6; s++){
        ctx.fillText(_tabTuningNames[s], leftPad - 7, topPad + s * stringSpacing + 4);
      }

      // Beat separator lines (dashed)
      var noteArea = barWidth - leftPad - rightPad - 4;
      for(var beat = 1; beat < maxBeat; beat++){
        var bx = leftPad + 4 + noteArea * (beat / maxBeat);
        ctx.setLineDash([2, 4]);
        ctx.beginPath();
        ctx.moveTo(bx, topPad - 4);
        ctx.lineTo(bx, topPad + 5 * stringSpacing);
        ctx.strokeStyle = 'rgba(255,255,255,0.05)';
        ctx.lineWidth = 0.5;
        ctx.stroke();
        ctx.setLineDash([]);
      }

      // Draw notes with chord-aware layout
      var notes = bar.notes || [];
      // Group notes by (string, beat) for overlap resolution
      var groups = {};
      notes.forEach(function(n){
        var beatKey = Math.round(n.beat * 10) / 10;
        var key = n.string + '_' + beatKey;
        if(!groups[key]) groups[key] = [];
        groups[key].push(n);
      });

      notes.forEach(function(n){
        var bx = leftPad + 4 + noteArea * (Math.min(n.beat, maxBeat - 0.01) / maxBeat);
        var by = topPad + (n.string - 1) * stringSpacing;
        var fret = n.fret;

        // Layout multiple notes at same (string, beat): fan outward
        var beatKey = Math.round(n.beat * 10) / 10;
        var groupKey = n.string + '_' + beatKey;
        var group = groups[groupKey];
        if(group && group.length > 1){
          var idx = group.indexOf(n);
          var total = group.length;
          // Spread from -8px to +8px per note, alternating
          var span = Math.min(total * 8, 20);
          var offset = -span/2 + (idx + 0.5) * (span / total);
          by += offset;
        }

        // Note dot (open strings smaller)
        ctx.beginPath();
        ctx.arc(bx, by, fret === 0 ? 2.5 : 5, 0, Math.PI * 2);
        ctx.fillStyle = fret === 0 ? 'rgba(255,255,255,0.12)' : 'rgba(244,143,177,0.55)';
        ctx.fill();

        // Note ring for non-zero frets
        if(fret > 0){
          ctx.beginPath();
          ctx.arc(bx, by, 5, 0, Math.PI * 2);
          ctx.strokeStyle = 'rgba(244,143,177,0.3)';
          ctx.lineWidth = 1;
          ctx.stroke();
        }

        // Fret number (skip if too many overlaps would be unreadable)
        if(!group || group.length <= 3){
          ctx.fillStyle = fret === 0 ? 'rgba(168,216,234,0.4)' : '#e0d0e8';
          ctx.font = (fret > 9 ? 'bold 9px' : 'bold 11px') + ' monospace';
          ctx.textAlign = 'center';
          ctx.fillText(fret === 0 ? '0' : String(fret), bx, by - 8);
        }
      });

      // Bar number
      ctx.fillStyle = '#3a2a50';
      ctx.font = '9px monospace';
      ctx.textAlign = 'center';
      ctx.fillText('' + (row.idx + bi + 1), barWidth / 2, barHeight - 5);

      rowDiv.appendChild(canvas);
    });

    wrap.appendChild(rowDiv);
  });
}
