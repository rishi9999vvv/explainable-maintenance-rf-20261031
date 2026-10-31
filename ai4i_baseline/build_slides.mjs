// Presentation content comes from executed results via build_materials.py.
// Invoke buildDeck inside artifactSession.run({artifactType:'presentation'}).
export function buildDeck(presentation, content) {
  const C={ink:'#153247',accent:'#087F8C',muted:'#526B7C',white:'#FFFFFF',bg:'#F7FAFC'};
  function text(slide,value,x,y,w,h,size=28,color=C.ink,bold=false){
    const sh=slide.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
    sh.text=value;sh.text.style={fontFamily:'Arial',fontSize:size,color,bold};return sh;
  }
  function table(slide,headers,rows,top=200,height=310){
    const t=slide.tables.add({rows:rows.length+1,columns:headers.length,left:72,top,width:1136,height,values:[headers,...rows]});
    t.cells.block({row:0,column:0,rowCount:rows.length+1,columnCount:headers.length}).assign({textStyle:{fontSize:24,typeface:'Arial',color:C.ink},fill:C.white});
    t.cells.block({row:0,column:0,rowCount:1,columnCount:headers.length}).assign({fill:C.ink,textStyle:{fontSize:24,bold:true,typeface:'Arial',color:C.white}});
    return t;
  }
  for(const [i,item] of content.entries()){
    const s=presentation.slides.add();s.background.fill=item.kind==='cover'?C.ink:C.bg;
    s.speakerNotes.text=item.notes;
    if(item.kind==='cover'){
      text(s,'Explainable AI-Based\nPredictive Maintenance',72,100,1136,190,64,C.white,true);
      text(s,item.lines[0],72,320,1100,100,36,'#88D8D7');
      text(s,item.lines[1],72,495,1100,52,27,C.white);
      text(s,item.lines[2],72,570,1100,70,24,'#CCDEE8');
      continue;
    }
    text(s,item.title,72,50,1136,100,44,C.ink,true);
    text(s,String(i+1).padStart(2,'0'),1150,660,58,28,18,C.muted);
    if(item.kind==='table'){
      const top=180,h=item.rows.length<=2?230:310;
      table(s,item.headers,item.rows,top,h);
      item.lines.forEach((v,j)=>text(s,v,72,top+h+38+j*58,1100,55,25,C.muted));
    } else if(item.kind==='chart'){
      s.charts.add('bar',{position:{left:95,top:175,width:1090,height:365},categories:item.categories,
        series:[{name:'Test average precision',values:item.values,fill:C.accent}],hasLegend:false,
        dataLabels:{showValue:true,position:'outEnd',textStyle:{fontSize:22,typeface:'Arial'},numberFormatCode:'0.000'},
        xAxis:{textStyle:{fontSize:22,typeface:'Arial'}},yAxis:{min:0,max:1,title:'Average precision',textStyle:{fontSize:20,typeface:'Arial'},majorGridlines:{fill:'#D8E3EA',width:1}}});
      text(s,item.lines[0],72,585,1100,65,25,C.muted);
    } else if(item.kind==='shapchart'){
      table(s,['Feature','Mean |SHAP|'],item.categories.map((c,k)=>[c,item.values[k].toFixed(4)]),175,365);
      text(s,item.lines[0],72,580,1100,70,25,C.muted);
    } else {
      const many=item.lines.length>4; const step=many?88:112;
      item.lines.forEach((v,j)=>text(s,v,72,178+j*step,1110,many?82:104,many?27:30));
    }
  }
}
