(this["webpackJsonpreact-fronend"]=this["webpackJsonpreact-fronend"]||[]).push([[0],{10:function(e,t,n){},11:function(e,t,n){"use strict";n.r(t);var c=n(1),i=n.n(c),s=n(0),o=n(3),a=n.n(o);n(9),n(10);var l=function(){return Object(s.jsxs)("div",{class:"main-container",children:[Object(s.jsx)("div",{class:"output-box",children:Object(s.jsx)("img",{id:"",alt:""})}),Object(s.jsxs)("div",{class:"wrapper",children:[Object(s.jsx)("img",{id:"input_image"}),Object(s.jsxs)("div",{class:"btn-container",children:[Object(s.jsxs)("label",{class:"custom-file-upload btn upload",children:[Object(s.jsx)("input",{type:"file",onChange:function(e){var t=new FileReader;t.onload=function(){document.getElementById("input_image").src=t.result,fetch("/result",{method:"POST",cache:"no-cache",headers:{"Content-type":"application/json"},body:JSON.stringify(t.result)}).then((function(e){return console.log(e.json())})).catch((function(e){console.log(e)}))},t.readAsDataURL(e.target.files[0])}}),"Upload"]}),Object(s.jsx)("input",{type:"button",class:"btn fil",value:"Filter"})]})]})]})},r=function(e){e&&e instanceof Function&&n.e(3).then(n.bind(null,12)).then((function(t){var n=t.getCLS,c=t.getFID,i=t.getFCP,s=t.getLCP,o=t.getTTFB;n(e),c(e),i(e),s(e),o(e)}))};a.a.render(Object(s.jsx)(i.a.StrictMode,{children:Object(s.jsx)(l,{})}),document.getElementById("root")),r()},9:function(e,t,n){}},[[11,1,2]]]);
//# sourceMappingURL=main.f0a90801.chunk.js.map