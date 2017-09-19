all_heroes = [];
no_mmr = true;
$("body").hide();
$('html').append('<i id="loader" style="position:absolute;top:45%;left:45%;font-size:128px;" class="fa fa-6x fa-circle-o-notch fa-spin" aria-hidden="true"></i>');
$(window).on("load", function() {
    $('#loader').remove();
    //Creates the selected hero containers,
    setTimeout(function(){
      $("#temp").remove();
      $("#radiant-heroes").append(`
        <div class='col-sm-2 col-sm-offset-1 well hero-well' style='padding:0px;' >
          <img style="" id='dire-puck' src='/static/images/puck_full.jpg'>
        </div>
        <div class='col-sm-2  well hero-well' style='padding:0px;' >
          <img style="" id='dire-puck' src='/static/images/puck_full.jpg'>
        </div>
        <div class='col-sm-2  well hero-well' style='padding:0px;' >
          <img style="" id='dire-puck' src='/static/images/puck_full.jpg'>
        </div>
        <div class='col-sm-2  well hero-well' style='padding:0px;' >
          <img style="" id='dire-puck' src='/static/images/puck_full.jpg'>
        </div>
        <div class='col-sm-2  well hero-well' style='padding:0px;' >
          <img style="" id='dire-puck' src='/static/images/puck_full.jpg'>
        </div>
      `);
      object = $("#radiant-heroes");
      object.children('.col-sm-2').css("min-width", object.find('img').width());
      object.children('.col-sm-2').css("min-height", object.find('img').height()+5);
      object.find('img').remove();
      $("#dire-heroes").append(`
        <div class='col-sm-2 col-sm-offset-1 well hero-well' style='padding:0px;' >
          <img style="" id='dire-puck' src='/static/images/puck_full.jpg'>
        </div>
        <div class='col-sm-2  well hero-well' style='padding:0px;' >
          <img style="" id='dire-puck' src='/static/images/puck_full.jpg'>
        </div>
        <div class='col-sm-2  well hero-well' style='padding:0px;' >
          <img style="" id='dire-puck' src='/static/images/puck_full.jpg'>
        </div>
        <div class='col-sm-2  well hero-well' style='padding:0px;' >
          <img style="" id='dire-puck' src='/static/images/puck_full.jpg'>
        </div>
        <div class='col-sm-2  well hero-well' style='padding:0px;' >
          <img style="" id='dire-puck' src='/static/images/puck_full.jpg'>
        </div>
      `);
      object = $("#dire-heroes");
      object.children('.col-sm-2').css("min-width", object.find('img').width());
      object.children('.col-sm-2').css("min-height", object.find('img').height()+5);
      object.find('img').remove();
    });
    $("body").show();
});
//Adds the heroes to the grid for choosing - sorted by faction > primary attribute > hero id.
for (var attr in radiant_heroes) {
  for (var i = 0; i < radiant_heroes[attr].length; i++){
    hero = ""+radiant_heroes[attr][i];
    all_heroes.push(hero);
    edited_hero = hero.replace(/ /g, '_').replace("\'", '').toLowerCase();
    edited_hero_2 = hero.replace('\'','.');
    $("#rad-"+attr+"-heroes").append(`
      <div class='col-sm-2 hero-selector'>
        <img class='grid-hero' id='`+edited_hero+`' src='/static/images/`+edited_hero+`_full.jpg' name='`+edited_hero_2+`'>
      </div>
    `);
  }
}
for (var attr in dire_heroes) {
  for (var i = 0; i < dire_heroes[attr].length; i++){
    hero = ""+dire_heroes[attr][i];
    all_heroes.push(hero);
    edited_hero = hero.replace(/ /g, '_').replace("\'", '').toLowerCase();
    edited_hero_2 = hero.replace('\'','.');
    $("#dire-"+attr+"-heroes").append(`
      <div class='col-sm-2 hero-selector'>
        <img class='grid-hero' id='`+edited_hero+`' src='/static/images/`+edited_hero+`_full.jpg' name='`+edited_hero_2+`'>
      </div>
    `);
  }
}

/**
 *   Checks to make sure cases have passed to allow the user to use the suggest or predict buttons.
 *   The cases include:
 *       1. - MMR field must be filled out (with a number 0 < x < 10000)
 *       2. - Either one of the following:
 *         2.1 - 9 heroes are picked (suggest button)
 *         2.2 - 10 heroes are picked (predict winner button)
 */
function info_validator(){
  //empty_string_check();
  if(!isNaN($("#mmr").val()) && $("#mmr").val() > 0 && $("#mmr").val() < 10000){
    no_mmr = false;
  }
  else{
    no_mmr = true;
  }
  if($("#selected-heroes").find(".hero").length == 9 && !no_mmr){
    $("#predict_button").addClass("disabled");
    $("#suggest_button").removeClass("disabled");
  }
  else if($("#selected-heroes").find(".hero").length == 10 && !no_mmr){
    $("#predict_button").removeClass("disabled");
    $("#suggest_button").addClass("disabled");
  }
  else{
    $("#predict_button").addClass("disabled");
    $("#suggest_button").addClass("disabled");
  }
}

/**
 * Keydown/keyup listener for the MMR input element.
 */
$('input').on('keydown keyup', function () {
  if(!isNaN($(this).val()) && $(this).val() > 0 && $(this).val() < 10000){
    no_mmr = false;
  }
  else{
    no_mmr = true;
  }
  info_validator();
});

/**
 * Removes any white spaces that were appended to the array unique_heroes after the user used the suggest_button.
 * @deprecated
 */
/*
function empty_string_check(){
  for(var i = 0; i < unique_heroes.length; i++){
    if(!(unique_heroes[i].length)){
      unique_heroes.splice(i, 1);
    }
  }
}
*/

/**
 * Sends data (heroes the user selected) to the Flask backend.
 */
function send_data(){
  unique_heroes = []
  $("#radiant-heroes").children(".col-sm-2").each(function(){
    if($(this).children('img').length > 0){
      unique_heroes.push($(this).children('img').attr("name").replace('.','\''));
    }
    else{
      unique_heroes.push("");
    }
  });
  $("#dire-heroes").children(".col-sm-2").each(function(){
    if($(this).children('img').length > 0){
      unique_heroes.push($(this).children('img').attr("name").replace('.','\''));
    }
    else{
      unique_heroes.push("");
    }
  });
  data = {}
  //console.log(unique_heroes);
  data['mmr'] = $("#mmr").val();
  data['heroes'] = unique_heroes;
  //post request to send data object to backend
  $.ajax({
		type : "POST",
		data: JSON.stringify(data, null, '\t'),
		contentType: 'application/json;charset=UTF-8',
    //if the post was successful, the text from the backend is returned
    success: function(text) {
      $("#data-modal").modal('toggle');
      $("#text_area").html(text);
    }
	});
}

/**
 * Mouseover/mouseleave listeners for the hero portraits that users can select.
 * Will add the 'Radiant/Dire' text onto the portraits on the bottomleft/top right of hero portrait.
 */
$(".hero-selector").on({
  mouseover: function () {
      if($(this).hasClass("hero-selector")){
        $(this).append(`
          <p class="rad-txt" >
            Radiant
          </p>
          <p class="dire-txt">
              Dire
          </p>
        `);
        $(this).find('p').css("visibility", "visible");
        $(this).find('p').css("font-size", $(this).children('img').width()/5.1+"px");
        $(this).find('p').css("pointer-events", "none");

        $(this).find('p.rad-txt').css("top", $(this).children('img').height()/2+"px");
        $(this).find('p.dire-txt').css("top", "0px");

        //$(this).find('p.dire-txt').css("padding-bottom", 10+"px");
        $(this).find('p').delay(0).animate({ opacity: 0.9 }, 500);

      }
      else{
        $(this).find('p').remove();
      }
  },
    mouseleave: function () {
      $(this).find('p').remove();
    }
});


/**
 * Onclick listener for the class 'grid-hero', which is the hero portraits in the grid.
 * Defines the functionality of users clicking (picking) a hero.
 */
$(document).on("click", ".grid-hero", function(e){
  if($(this).parent(".col-sm-2").hasClass("hero-picked")){
    return
  }
  //clear the search bar if a hero is picked
  search_clear();
  //manipulating strings for ids of divs and what not
  name = $(this).attr('name').replace('.', '\'');
  name_original = $(this).attr('name');
  edited_name = name.replace(/ /g, '_').replace("\'", '').toLowerCase();
  var pos = $(this).position();
  //The following are the x/y coordinates of the mouse click relative to image.
  var x = e.pageX - pos.left;
  var y = e.pageY - pos.top;
  //Width and height of the div of the image clicked
  var width = $(this).width();
  var height = $(this).height();
  //location the user clicked on the image. subtract x (location of their click on the entire screen) by the offset/location of the image
  click_location_x = x - $(this).offset()['left']
  click_location_y = y - $(this).offset()['top']
  //get the slope of the triangle drawn on the image (which is just height/width)
  slope = (height)/(width);

  //if there are 10 heroes and user tries to add, do nothing
  if( $("#selected-heroes").find(".hero").length == 10){
  }
  //else if they clicked on dire's side, the hero wasn't already pciked and there are less than 5 dire heroes already picked, OR
  //already 5 radiant heroes, the hero wasn't already picked and there are less than 5 dire heroes already picked
  //then we add the hero to dire
  //took the above out: || $("#radiant-heroes").children().length == 5 && !hero_exist_check(edited_name) && $("#dire-heroes").children().length < 5
  else if(click_location_x*slope > click_location_y && !hero_exist_check(edited_name) && $("#dire-heroes").children('.hero').length < 5){
    $("#dire-heroes").children('.well').each(function(){
      if($(this).children('img').length == 0){
        $(this).addClass("dire-hero");
        $(this).addClass("hero");
        $(this).append(`
          <img name='`+name_original+`' id='dire-`+edited_name+`' src='/static/images/`+edited_name+`_full.jpg'>
        `);
        return false
      }
    });
    //add classes to gray it out and remove classes so it doesnt show the green/red overlay if its picked
    $(this).parent(".col-sm-2").addClass("hero-picked");
    if($(this).parent(".col-sm-2").hasClass("hero-selector")){
      $(this).parent(".col-sm-2").removeClass("hero-selector");
      $(this).parent(".col-sm-2").addClass("hero-selector-2");
      $(this).parent().find('p').remove();
    }
  }
  //else if click loc was under the line and there are < 5 radiant heroes and the hero wasn't already picked
  //then we add the hero to radiant
  else if(click_location_x*slope <= click_location_y && $("#radiant-heroes").children('.hero').length < 5 && !hero_exist_check(edited_name)){
    //below cases are just for the offset to add to the div to make it centered
    $("#radiant-heroes").children('.well').each(function(){
      if($(this).children('img').length == 0){
        $(this).addClass("radiant-hero");
        $(this).addClass("hero");
        $(this).append(`
          <img name='`+name_original+`' id='radiant-`+edited_name+`' src='/static/images/`+edited_name+`_full.jpg'>
        `);
        return false
      }
    });
    //add classes to gray it out and remove classes so it doesnt show the green/red overlay if its picked
    if($(this).parent(".col-sm-2").hasClass("hero-selector")){
      $(this).parent(".col-sm-2").removeClass("hero-selector");
      $(this).parent(".col-sm-2").addClass("hero-selector-2");
      $(this).parent(".col-sm-2").addClass("hero-picked");
      $(this).parent().find('p').remove();
    }
  }
  //else do nothing (maybe something will happen here in the future)
  else{
  }
  info_validator();
});

/**
 * Function that checks if a hero was already selected.
 * @param {String} edited_name - edited hero name (hero name with replaced characters such as spaces and ')
 * @returns {Number} - 0 if hero hasn't already been chosen, or  > 0 if it has
 */
function hero_exist_check(edited_name){
  return $("#dire-"+edited_name).length + $("#radiant-"+edited_name).length
}

/**
 * Onclick listener for heroes that are already chosen.
 * Will remove them from the selected hero pool if clicked.
 */
$(document).on("click", ".hero", function(e){
  //was naming id's like "dire-bane", "rad-luna" but anti-mage giving error with split("-")[1], so now removing split("-")[0] and then joining the array elements
  //should probably change this to regex
  element_id = $(this).children("img").attr("id").split("-");
  element_id.splice(0, 1);
  element_id=element_id.join().replace(",", "-");

  //checks if it has the class first. fixes issue with search bar that resulted in nongray portraits if hero is removed
  if($("#"+element_id).parent(".col-sm-2").hasClass("hero-picked") &&  (~$("#"+element_id).attr('id').indexOf($("#hero-search").val()))){
    $("#"+element_id).parent(".col-sm-2").removeClass("hero-picked");
    $("#"+element_id).parent(".col-sm-2").removeClass("hero-selector-2");
    $("#"+element_id).parent(".col-sm-2").addClass("hero-selector");
  }
  //remove all relevant classes, image and icons from the well
  $(this).removeClass($(this).children().attr('id').split("-")[0]+"-hero");
  $(this).removeClass("hero");
  $(this).children('img').remove();
  $(this).children('i').remove();
  info_validator();
});

/**
 * Onclick listener for the buttons.
 */
$(document).on("click", ".submit-btn", function(e){
  $(this).tooltip();
  if(!$(this).hasClass('disabled')){
    send_data();
  }
});


/**
 * Onclick listener for the search bar.
 */
 /*
$(document).on("click", "#hero-search", function(e){
  $('.grid-hero').each(function( index ) {
    if(~$(this).attr('id').indexOf($("#hero-search").val()) && !hero_exist_check($(this).attr('id'))){
      $(this).parent(".col-sm-2").removeClass("hero-picked");
      $(this).parent(".col-sm-2").removeClass("hero-selector-2");
      $(this).parent(".col-sm-2").addClass("hero-selector");
    }
    else if(!$(this).parent(".col-sm-2").hasClass("hero-picked")){
      $(this).parent(".col-sm-2").removeClass("hero-selector");
      $(this).parent(".col-sm-2").removeClass("hero-selector-2");
      $(this).parent(".col-sm-2").removeClass("hero-picked");
      $(this).parent(".col-sm-2").addClass("hero-selector-2");
      $(this).parent(".col-sm-2").addClass("hero-picked");
    }
    else{}
  });
});
*/

/**
 * Deselect/blur listener for the search bar.
 * @deprecated
 */
/*
$(document).on("blur", "#hero-search", function(e){
  $('.grid-hero').each(function( index ) {
    if(!hero_exist_check($(this).attr('id'))){
      $(this).parent(".col-sm-2").removeClass("hero-picked");
      $(this).parent(".col-sm-2").removeClass("hero-selector-2");
      $(this).parent(".col-sm-2").addClass("hero-selector");
    }
    else if(!$(this).parent(".col-sm-2").hasClass("hero-picked")){
      $(this).parent(".col-sm-2").removeClass("hero-selector");
      $(this).parent(".col-sm-2").removeClass("hero-selector-2");
      $(this).parent(".col-sm-2").removeClass("hero-picked");
      $(this).parent(".col-sm-2").addClass("hero-selector-2");
      $(this).parent(".col-sm-2").addClass("hero-picked");
    }
    else{}
  });
});
*/

/**
 * Auto complete search bar.
 * Imported from: https://goodies.pixabay.com/javascript/auto-complete/demo.html
 */
all_heroes = all_heroes.sort();
var my_autoComplete = new autoComplete({
    selector: 'input[name="hero-search"]',
    minChars: 1,
    source: function(term, suggest){
      term = term.toLowerCase();
      var choices = all_heroes;
      var matches = [];
      for (i=0; i<choices.length; i++)
          if (~choices[i].toLowerCase().indexOf(term)) {
            matches.push(choices[i]);
          }
      if(matches.length <= 0){
        $('.grid-hero').each(function( index ) {
          $(this).parent(".col-sm-2").removeClass("hero-selector");
          $(this).parent(".col-sm-2").removeClass("hero-selector-2");
          $(this).parent(".col-sm-2").removeClass("hero-picked");
          $(this).parent(".col-sm-2").addClass("hero-selector-2");
          $(this).parent(".col-sm-2").addClass("hero-picked");
        });
      }
      suggest(matches);
    },
    renderItem: function (item, search){
        edited_name = item.replace(/ /g, '_').replace("\'", '').toLowerCase();
        search = search.replace(/ /g, '_').replace("\'", '').toLowerCase();
        var re = new RegExp("(" + search.split(' ').join('|') + ")", "gi");
        $('.grid-hero').each(function( index ) {
          if(~$(this).attr('id').indexOf(search) && !hero_exist_check($(this).attr('id'))){
            $(this).parent(".col-sm-2").removeClass("hero-picked");
            $(this).parent(".col-sm-2").removeClass("hero-selector-2");
            $(this).parent(".col-sm-2").addClass("hero-selector");
          }
          else if(!$(this).parent(".col-sm-2").hasClass("hero-picked") && !(~$(this).attr('id').indexOf(search))){
            $(this).parent(".col-sm-2").removeClass("hero-selector");
            $(this).parent(".col-sm-2").addClass("hero-selector-2");
            $(this).parent(".col-sm-2").addClass("hero-picked");
          }
          else{
            $(this).parent(".col-sm-2").removeClass("hero-selector");
            $(this).parent(".col-sm-2").addClass("hero-selector-2");
            $(this).parent(".col-sm-2").addClass("hero-picked");
          }
        });
        if(hero_exist_check(edited_name) || edited_name == search){
          return ''
        }
        return `
        <div class="autocomplete-suggestion" data-val="`+item+`">
          <img style="pointer-events:none;width:25px;height:25px;float:left;" src="/static/images/miniheroes/`+edited_name+`.png">
          <p style="float:left;padding-left:10px;">  `+item+`</p>
        </div>
          `
    },
});

/**
 * Clears text that is in the search bar. Also resets all gray'd out heroes if they aren't selected.
 */
function search_clear(){
  $('#hero-search').val('');
  $('.grid-hero').each(function( index ) {
    if(!hero_exist_check($(this).attr('id'))){
      $(this).parent(".col-sm-2").removeClass("hero-picked");
      $(this).parent(".col-sm-2").removeClass("hero-selector-2");
      $(this).parent(".col-sm-2").addClass("hero-selector");
    }
    else if(!$(this).parent(".col-sm-2").hasClass("hero-picked")){
      $(this).parent(".col-sm-2").removeClass("hero-selector");
      $(this).parent(".col-sm-2").removeClass("hero-selector-2");
      $(this).parent(".col-sm-2").removeClass("hero-picked");
      $(this).parent(".col-sm-2").addClass("hero-selector-2");
      $(this).parent(".col-sm-2").addClass("hero-picked");
    }
    else{}
  });
}

/**
 * Mouse over listener for hero class (selected hero).
 */
$(document).on("mouseover", ".hero", function(e){
  $(this).append(`
    <i style="color:red;position:absolute;top:0%;right:5%;text-shadow: 1px 1px 1px #000;" class="fa fa-2x fa-times" aria-hidden="true"></i>
  `);
});

/**
 * Mouse leave listener for hero class (selected hero).
 */
$(document).on("mouseleave", ".hero", function(e){
  $(this).find('i').remove();
});

/**
 * Mouse leave listener for buttons.
 */
$(document).on("mouseleave", "button", function(e){
  $(this).tooltip();
});

/**
 * Make sure radiant/dire text at top matches each other's widht/height
 */
 function setHeight(elem1, elem2) {
  var height = elem2.height()
  elem1.css('height', height);
}

$(document).ready(function() {

//setHeight($('#dire-text-h1'), $('#rad-text-h1'));

// When the window is resized the height might
// change depending on content. So to be safe
// we rerun the function
$(window).on('resize', function() {
    setHeight($('#dire-text-h1'), $('#rad-text-h1'));
});

});
