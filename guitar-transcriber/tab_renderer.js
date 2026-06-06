// Guitar Tab Canvas Renderer
// Renders tab data (bars/notes) onto Canvas elements
var _tabTuningNames = ['e','B','G','D','A','E'];

function renderTab(data){
  var bars = data.bars || [];
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
  var topPad = 24;
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
      canvas.style.cssText += 'background:rgba(255,255,255,0.02);border-radius:8px;border:1px solid rgba(255,255,255,0.04)';

      var ctx = canvas.getContext('2d');
      ctx.scale(dpr, dpr);

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

      // Draw notes
      var notes = bar.notes || [];
      var drawnPositions = {};  // track drawn positions to avoid overlap

      notes.forEach(function(n){
        var bx = leftPad + 4 + noteArea * (Math.min(n.beat, maxBeat - 0.01) / maxBeat);
        var by = topPad + (n.string - 1) * stringSpacing;
        var fret = n.fret;

        // Offset overlapping notes on same string+beat
        var posKey = n.string + '_' + Math.round(n.beat * 10) / 10;
        var offsetCnt = drawnPositions[posKey] || 0;
        drawnPositions[posKey] = offsetCnt + 1;
        if(offsetCnt > 0){
          by -= offsetCnt * 7;
        }

        // Note dot
        ctx.beginPath();
        ctx.arc(bx, by, fret === 0 ? 2.5 : 5.5, 0, Math.PI * 2);
        ctx.fillStyle = fret === 0 ? 'rgba(255,255,255,0.12)' : 'rgba(244,143,177,0.55)';
        ctx.fill();

        // Note ring for non-zero frets
        if(fret > 0){
          ctx.beginPath();
          ctx.arc(bx, by, 5.5, 0, Math.PI * 2);
          ctx.strokeStyle = 'rgba(244,143,177,0.3)';
          ctx.lineWidth = 1;
          ctx.stroke();
        }

        // Fret number
        ctx.fillStyle = fret === 0 ? 'rgba(168,216,234,0.4)' : '#e0d0e8';
        ctx.font = (fret > 9 ? 'bold 9px' : 'bold 11px') + ' monospace';
        ctx.textAlign = 'center';
        ctx.fillText(fret === 0 ? '0' : String(fret), bx, by - 8);
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
