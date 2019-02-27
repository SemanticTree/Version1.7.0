var container_id="map-area";
var cy, display_cy;
var style=[
	{
		"selector":"node",
		"style" : {
			"label":"data(title)",
			"text-valign":"center",
			'text-rotation':'autorotate',
			'font-size' : '2em',
			'padding':'15px',
			'height' : 'label',
			'width' : 'label',
		}
	},{
	  	"selector":"edge",
	  	"style" : {
			'curve-style':'bezier',
			'font-size': '2em',
      // 'target-arrow-shape' : 'triangle',
      // 'arrow-scale' : 2,
      // // Orignal ----------------------
			'text-rotation':'autorotate',
			'label' : 'data(title)',
		}
  },{
			"selector":"[?is_central]",
			"style":{
				"background-color" : "#036169",
				"color" : "white",

			}
		}];

let cxtmenuDefaults = {
  menuRadius: 100, // the radius of the circular menu in pixels
  selector: 'node', // elements matching this Cytoscape.js selector will trigger cxtmenus
  commands: [ // an array of commands to list in the menu or a function that returns the array
  	// {
  	// 	content: 'ChangeColor',
  	// 	select: () => randomColor(),
  	// 	enabled: true
  	// },
  	{
  		content: 'Origin Text',
  		select: (ele) =>  showText(ele),
  	},{
  		content: 'Get Subtree',
  		select: function(ele){
  			addStyle(level_limit, ele);
  		}
  	},
  ], // function( ele ){ return [ /*...*/ ] }, // example function for commands
  fillColor: 'rgba(0, 0, 0, 0.75)', // the background colour of the menu
  activeFillColor: 'rgba(125, 125, 125, 0.75)', // the colour used to indicate the selected command
  activePadding: 10, // additional size in pixels for the active command
  indicatorSize: 24, // the size in pixels of the pointer to the active command
  separatorWidth: 3, // the empty spacing in pixels between successive commands
  spotlightPadding: 4, // extra spacing in pixels between the element and the spotlight
  minSpotlightRadius: 24, // the minimum radius in pixels of the spotlight
  maxSpotlightRadius: 38, // the maximum radius in pixels of the spotlight
  openMenuEvents: 'taphold', // space-separated cytoscape events that will open the menu; only `cxttapstart` and/or `taphold` work here
  itemColor: 'white', // the colour of text in the command's content
  itemTextShadowColor: 'transparent', // the text shadow colour of the command's content
  zIndex: 9999, // the z-index of the ui div
  atMouse: false // draw menu at mouse position
};

function levelColor(level) {
	green_color = [ "#036169","#045659","#588c8f","#96bec0","#edf4f5"];
	red_color = [ "#742344","#A53067","#B43D75","#DB5791","#F19ECC"];
	blue_green_color = [ "#0D59FF","#0C8BE8","#00D7FF","#0CE8CE","#0DFF9E"];
	if (level < level_limit){
		return red_color[level];
	} else {
		return "rgb(135,135,135)";
	}
}
var level_limit=2;
var responseData;
