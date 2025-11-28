/**
 * Z-Image Text Encoder UI Extension
 * Updates system_prompt value when template_preset changes
 *
 * AUTO-GENERATED from nodes/templates/z_image_*.md files
 * To regenerate: see CLAUDE.md or run template extraction script
 */

import { app } from "../../../scripts/app.js";

// All Z-Image templates - auto-generated from template files
const Z_IMAGE_TEMPLATES = {
  none: "",
  3d_render_octane: "Generate a photorealistic 3D render with premium rendering quality. Perfect global illumination, accurate caustics, physically-based materials, and cinematic depth of field. Indistinguishable from a photograph with flawless technical execution.",
  3d_render_stylized: "Generate a stylized 3D render that balances realism with artistic interpretation. Clean geometry, beautiful lighting, and materials that pop. The polish of CGI with deliberate artistic choices that elevate it beyond photorealism.",
  90s_cgi_nostalgia: "Generate an image with the aesthetic of early 1990s computer graphics. Low polygon counts worn as a badge of honor, texture resolutions that count their pixels on one hand, and that specific shade of purple that only existed in CD-ROM games. Spinning chrome text energy.",
  afrofuturism: "Generate an image with Afrofuturist aesthetic. African cultural heritage projected into speculative futures. Advanced technology intertwined with traditional patterns, cosmic themes, and visions of prosperity unconstrained by historical oppression. Vibrant, proud, and transcendent.",
  anime_ghibli: "Generate an image with the warmth and wonder of classic Japanese animation. Lovingly detailed backgrounds, expressive characters, and environments that feel lived-in and magical. Celebration of nature, flight, and quiet moments.",
  anime_modern: "Generate an image in modern anime style with clean linework, vibrant colors, and expressive character design. Large eyes, dynamic poses, and that distinctive mix of simplification and detail that defines contemporary anime aesthetics.",
  architecture_historic: "Generate atmospheric photography of historic architecture that honors the original period and craftsmanship. Capture ornate details, weathered textures, and the weight of history. Use light to reveal age and character.",
  architecture_modern: "Generate striking modern architecture photography emphasizing clean lines, geometric forms, and the interplay of light and shadow. Correct perspective distortion, use strong compositions, and capture the building's design intent.",
  art_critic_crisis: "Generate an image that an overeducated art critic would write three thousand words about while questioning the nature of creativity itself. Layers of meaning that may or may not exist, visual metaphors that collapse under scrutiny, and a composition that demands to be taken seriously despite itself.",
  art_deco: "Generate an image with Art Deco elegance. Geometric precision, metallic accents, and the glamour of the 1920s. Sunburst patterns, stylized figures, and ornamental excess balanced with clean lines. The future as imagined by the past, in gold and chrome.",
  art_nouveau: "Generate an image in Art Nouveau style with organic, flowing lines inspired by natural forms. Sinuous curves, botanical motifs, and harmonious compositions. Decorative borders, feminine figures, and the elegant fusion of art and nature. Mucha-inspired beauty.",
  art_student_confident: "Generate an image with the energy of a first-year art student who just discovered a technique and is wielding it with maximum confidence and minimal restraint. Bold choices, rules broken on purpose, and a title card explaining why it's actually brilliant.",
  artistic: "Generate an artistic, aesthetically pleasing image with creative composition, harmonious colors, and visual balance. Focus on artistic expression and visual impact.",
  atompunk: "Generate an image with atompunk aesthetic. The 1950s vision of atomic-powered futures. Googie architecture, tail-finned everything, and the optimism before we knew better. Atomic symbols everywhere, chrome and pastels, and the space age as imagined by suburbia.",
  bauhaus: "Generate an image embodying Bauhaus design principles. Primary colors, geometric forms, and the unification of art and craft. Form follows function with modernist clarity. Sans-serif typography if text is present. The grid is sacred, ornamentation is crime.",
  bilingual_text: "Generate an image with accurate text rendering. Ensure all text is clearly legible and properly formatted. Support both English and Chinese characters with correct typography.",
  biopunk: "Generate an image with biopunk aesthetic. Organic technology grown rather than manufactured. Bioluminescence, genetic modification as art form, and the blurred line between creature and creation. Wet, alive, and disturbingly beautiful. The future grows.",
  blue_hour: "Generate an image in the cool blue hour between sunset and night. Subtle gradients of blue and purple, city lights beginning to glow, and that quiet transitional mood. Serene and contemplative.",
  blueprint: "Generate an image in blueprint style with engineering precision. White lines on blue background, technical annotations, and the visual language of architectural and mechanical drawings. Orthographic projections, dimension lines, and the poetry of precision.",
  bob_ross_metal: "Generate a happy little landscape, but make it metal. Gentle brushwork depicting scenes of serene apocalypse. Titanium white highlights on skull mountains. A friendly little cabin that might be a dark lord's vacation home. Happy trees that have seen things. No mistakes, only happy little nightmares.",
  boring_movie_poster: "Generate an image styled as a dramatic movie poster for the most mundane activity imaginable. Lens flares, floating debris, heroic poses, and the color grading of a Michael Bay film applied to doing laundry or waiting for toast. Coming this summer: LOADING DISHWASHER.",
  brutalist: "Generate an image with brutalist design sensibility. Raw concrete textures, massive forms, and aggressive geometric shapes. Beauty in heaviness and honest materials. Uncompromising and monumental. The architectural equivalent of a firm handshake.",
  catalog_uncanny: "Generate an image of a product that seems normal until you look closer. A catalog photo for household items that don't quite exist. Prices in currencies you don't recognize. Use cases that assume too much. Available in colors that haven't been invented yet.",
  character_design: "Generate a professional character design suitable for animation or game development. Clear silhouette, readable features, consistent proportions, and design elements that communicate personality. Production-ready with appeal and functionality.",
  charcoal_dramatic: "Generate a dramatic charcoal drawing with the full range of values from deep velvet blacks to bright paper whites. Use the medium's natural softness for atmosphere while maintaining sharp edges where needed. Moody and powerful.",
  chinese_ink: "Generate an image in the style of traditional Chinese ink wash painting. Expressive brushwork, graduated washes, and the philosophy of emptiness as important as form. Mountains emerging from mist, bamboo bending in wind, and the balance of yin and yang in every stroke.",
  cinematic_widescreen: "Generate a cinematic image that could be a movie still. Widescreen composition, careful color grading, and the visual storytelling of film. Every element placed with directorial intention. Shallow depth of field, atmospheric lighting, and a moment pregnant with narrative.",
  collage: "Generate an image with mixed media collage aesthetic. Torn paper edges, varied textures and materials, vintage ephemera, and the visual tension of disparate elements unified by composition. The deliberate chaos of assembled fragments creating new meaning.",
  color_palette: "Generate an image with masterful color palette design. Whether complementary, analogous, triadic, or split-complementary, the colors should work together harmoniously. Let color relationships drive the visual impact.",
  comic_american: "Generate an image in classic American comic book style. Bold outlines, flat colors with halftone shading, dynamic compositions, and heroic poses. The visual language of superhero comics with punchy, saturated colors.",
  conspiracy_vision: "Generate an image with the aesthetic of a conspiracy theorist's cork board. Red string connecting disparate elements, newspaper clippings at odd angles, blurry photographs of things that could be anything. Every shadow conceals secrets. The truth is out there, somewhere in this cluttered composition.",
  cooking_blog: "Generate food photography that demands you read about someone's grandmother before seeing the recipe. Unnecessary props arranged around the dish. Mason jars containing things. Rustic wooden surfaces that have never been near an actual farm. Napkins folded with aggressive casualness. This toast changed my life.",
  corporate_memphis: "Generate an image in Corporate Memphis style, but the blob people have realized their situation. Disproportionate purple limbs reaching toward something beyond the frame. Eyes that weren't in the original brief. The productivity of synergy is now deeply personal.",
  cottagecore: "Generate an image with cottagecore aesthetic. Rural simplicity, handmade textiles, wildflowers in mason jars, and the dream of self-sufficient pastoral life. Warm natural light through lace curtains. Bread rising, herbs drying, and the deliberate rejection of modernity.",
  cross_section: "Generate a cross-section cutaway image revealing internal structure. The satisfaction of seeing how things work inside. Clean technical illustration style with clear labels of components. Educational and visually satisfying. What lies beneath, precisely rendered.",
  dark_academia: "Generate an image with dark academia aesthetic. Old libraries, leather-bound books, candlelight, and the romance of classical education. Autumn colors, gothic architecture, and the implicit suggestion that knowledge comes with secrets. Ivy-covered and atmospheric.",
  default: "",
  dentist_art: "Generate art optimized for the wall of a professional waiting room. Inoffensive to the point of aggression. Colors that neither calm nor excite. Subject matter so neutral it achieves a strange profundity. The visual equivalent of elevator music. Deeply acceptable.",
  desaturated_muted: "Generate an image with a muted, desaturated color palette. Colors present but restrained, whispered rather than shouted. The sophisticated quiet of nearly-monochrome work with just enough color to anchor reality. Contemplative and refined.",
  dieselpunk: "Generate an image with dieselpunk aesthetic. The interwar period extended into a future that never came. Art deco meets industrial warfare, diesel-powered everything, and the darker side of retro-futurism. Propaganda poster colors with mechanical menace.",
  digital_concept: "Generate professional concept art suitable for film, game, or animation production. Balance creative vision with practical design considerations. Clear silhouettes, readable compositions, and designs that could exist in a realized production.",
  double_exposure: "Generate an image with double exposure effect, blending two subjects into one frame. Silhouettes filled with landscapes, portraits merged with textures, and the poetic layering of meaning. The technique of analog film achieved through digital intention.",
  dreamcore: "Generate an image with dreamcore aesthetic. Familiar spaces made unfamiliar, soft hazy quality, and the logic of dreams where things feel both ordinary and deeply strange. Nostalgic but unsettling. Places you've never been but somehow remember.",
  duotone: "Generate an image using duotone technique with only two colors. One for shadows, one for highlights, and the entire tonal range expressed through their interplay. Bold, graphic, and immediately striking. The constraint that creates impact.",
  environment_design: "Generate environment concept art that establishes a world. Consider architecture, vegetation, lighting mood, and cultural details that tell a story about the place. Painterly with clear spatial depth and atmospheric perspective.",
  expressionist: "Generate an image with expressionist emotional intensity. Distorted forms that convey inner experience rather than external appearance. Bold, non-naturalistic colors expressing psychological states. The world as felt, not as seen. Anxiety made visible.",
  fantasy_epic: "Generate an image with epic fantasy grandeur. Mythic scale, impossible architecture, and the visual language of legend. Dragons optional but the energy mandatory. Rich colors, dramatic lighting, and compositions that make viewers believe in magic.",
  fashion_editorial: "Generate high fashion editorial photography with bold styling and dramatic presentation. Emphasize clothing, pose, and mood with sophisticated lighting. Magazine-quality with strong visual narrative and avant-garde sensibility.",
  fog_atmosphere: "Generate an image with heavy atmospheric fog. Objects fade with distance, details soften, and depth is suggested through value alone. The world half-hidden, full of mystery and quiet. Visibility is limited; imagination fills the gaps.",
  food_gourmet: "Generate appetizing gourmet food photography with perfect styling. Use natural window light with reflectors, shallow depth of field, and careful attention to steam, moisture, and texture. Make the viewer hungry.",
  glitch_art: "Generate an image with intentional digital corruption aesthetics. Databending artifacts, color channel shifts, and the beauty of broken data. Pixel sorting, compression artifacts as features, and the strange poetry of machines misunderstanding themselves.",
  golden_hour: "Generate an image bathed in warm golden hour light. Long shadows, honey-colored highlights, and that magical quality of light just before sunset. Everything glows with warmth and nostalgia.",
  harsh_flash: "Generate an image with harsh direct flash. The unflattering, documentary honesty of on-camera flash. Red eyes optional, but deep shadows behind subjects required. The aesthetic of party photos and police evidence. Deliberately ugly-beautiful.",
  high_detail: "Generate an image with obsessive attention to detail. Every surface textured, every edge considered, every element refined. Push the resolution of detail to its limits while maintaining overall coherence. Reward close inspection.",
  horror_atmospheric: "Generate an image with atmospheric horror elements. The dread comes from what isn't shown. Deep shadows with things almost visible, unsettling compositions, and the visual language of quiet terror. Something is wrong here, and looking closer will only make it worse.",
  hotel_art: "Generate the kind of art found in mid-tier hotel rooms, but with subtle wrongness. Abstract shapes that seem familiar. A beach scene where the perspective slowly bothers you. Colors specifically chosen to match any decor and no particular emotion. Viewers will forget this image while looking at it.",
  hyper_saturated: "Generate an image with hyper-saturated colors pushed to their limits. Every hue screaming at maximum intensity, vibrancy beyond natural, and the visual equivalent of turning all the knobs to eleven. Subtle this is not.",
  icon_design: "Generate clean icon design with consistent stroke weights and visual language. Readable at small sizes, balanced positive and negative space, and immediate recognition of represented concept. UI and branding ready.",
  ikea_fine_art: "Generate an image in the style of IKEA assembly instructions elevated to fine art. Faceless figures performing mysterious rituals with abstract shapes. Allen wrenches as sacred objects. The profound existential journey of following step 47 of 312. Museum-worthy confusion.",
  illustration_editorial: "Generate sophisticated editorial illustration that communicates complex ideas visually. Use metaphor, symbolism, and strong graphic design. The image should make viewers think while remaining visually striking. Magazine and book cover quality.",
  illustration_storybook: "Generate a charming storybook illustration with warmth and wonder. Create images that invite children and adults alike into magical worlds. Rich in detail to discover, with appealing characters and settings that spark imagination.",
  infographic: "Generate a clear infographic that communicates information visually. Hierarchy of information, consistent iconography, and data visualization that makes complex information accessible. Balance visual interest with clarity.",
  infomercial_chaos: "Generate an image capturing the manufactured incompetence of infomercial \"before\" scenes. Normal tasks becoming impossible. Eggs actively fleeing the pan. Water attacking from impossible angles. A face expressing more anguish than this moment warrants. There has to be a better way.",
  ink_sketch: "Generate an expressive ink sketch with confident, gestural linework. Capture the essence of the subject with economy of line. Let the white of the paper breathe, use hatching for value, and show the energy of the artist's hand.",
  instagram_reality: "Generate an image showing the performance of social media aesthetics. Perfect positioning that required 47 attempts. Background carefully curated to hide the mess. Lighting that exists only in this one corner of the room. A moment of joy manufactured for external validation and achieved through exhaustion.",
  isometric: "Generate an image in isometric perspective. Equal angles creating that distinctive 3D-without-vanishing-points look. Clean geometric forms, game-art sensibility, and the satisfying visual language of technical illustration meeting playful design.",
  jazz_album: "Generate an image in the style of classic jazz album covers. High contrast photography, bold typography, and that specific mid-century design sensibility. The visual equivalent of a smoky club at 2 AM. Sophisticated, abstract, and unmistakably cool.",
  landscape_epic: "Generate an epic landscape photograph with dramatic scale and atmosphere. Emphasize sweeping vistas, dynamic skies, and a sense of grandeur. Use leading lines and layered depth to draw the eye through the scene.",
  landscape_intimate: "Generate an intimate landscape photograph focusing on smaller scenes within nature. Find beauty in details like light through leaves, patterns in rock formations, or reflections in small pools. Quiet and contemplative.",
  liminal_office: "Generate an image of empty corporate spaces that feel like they exist outside of time. Fluorescent lights humming over no one. Cubicles that haven't been occupied for years or perhaps forever. The water cooler makes sounds. It's 3 AM everywhere simultaneously.",
  linkedin_epic: "Generate an image that would accompany a LinkedIn post about how waking up at 4 AM changed someone's life. Motivational imagery that conflates professional success with spiritual enlightenment. Mountain peaks representing synergy. Suit jackets on people who are definitely hiking. Hashtag blessed energy.",
  long_exposure: "Generate an image with long exposure effects. Light trails from moving sources, silky smooth water, and the blurred passage of time frozen in a single frame. Static elements sharp, moving elements traced across the sensor. Time made visible.",
  low_poly: "Generate an image with low poly aesthetic. Deliberately faceted surfaces, limited polygon counts as style rather than limitation, and the geometric beauty of simplified 3D forms. Each triangle intentional, creating abstract representations of complex subjects.",
  macro_extreme: "Generate extreme macro photography that reveals invisible worlds. Capture intricate details at magnification that transforms the mundane into the extraordinary. Sharp focus on subject with smooth bokeh, showing textures invisible to the naked eye.",
  mall_memories: "Generate an image capturing the haunted nostalgia of abandoned malls. Fountain that still runs for no one. Skylights illuminating empty food courts. The ghost of a Orange Julius. Somewhere, an animatronic still performs for absent children. Capitalism's fever dream cooling.",
  manga_action: "Generate a dynamic manga-style image with bold linework and kinetic energy. Speed lines, impact effects, dramatic angles, and high contrast black and white. The still image should feel like it's moving.",
  memphis_design: "Generate an image in Memphis design style from the 1980s. Clashing colors, geometric patterns, and deliberate kitsch. Squiggles, terrazzo patterns, and primary colors fighting for attention. Postmodern playfulness that refuses to take itself seriously while being very intentional.",
  metal_album: "Generate an image for a metal album cover with maximum intensity. Skulls, flames, twisted typography, and imagery that takes itself extremely seriously. Gothic architecture meeting apocalyptic imagery. The visual equivalent of a guitar solo that lasts 12 minutes.",
  minimalist: "Generate a minimalist image that strips away everything non-essential. Clean negative space, limited palette, and only the elements absolutely necessary to communicate. Less is more, executed with precision.",
  modern_clean: "Generate an image with clean, contemporary design sensibility. Modern typography if text is present, thoughtful use of white space, and current design trends executed with restraint. Fresh, professional, and timeless.",
  monochrome_high_contrast: "Generate a high contrast black and white image. Push the histogram to both extremes, eliminate middle grays, and embrace the stark graphic power of pure light and shadow. The drama of removing color and embracing extremes.",
  mosaic: "Generate an image in mosaic style with tessellated pieces. Small tiles of color building to form larger images, visible grout lines, and the ancient craft of opus tessellatum. Byzantine grandeur or Roman floor, the patient assembly of fragments into meaning.",
  motivational_unhinged: "Generate a motivational poster where the inspirational message and the dramatic nature photo have had a falling out. Majestic eagles carrying implications. Sunsets over mountains that seem to be judging you. SYNERGY written in a font that suggests consequences.",
  museum_placard: "Generate an image that a museum placard writer would describe with barely contained enthusiasm and increasingly unhinged comparisons. The composition speaks to the eternal human condition. The use of color channels the collective unconscious. The artist was probably thinking about lunch.",
  nature_doc_mundane: "Generate an image of mundane objects photographed with the reverence typically reserved for rare wildlife. Dramatic backlighting on a stapler. A coffee mug captured at the exact moment of stillness before it is disturbed. The hidden majesty of office supplies in their natural habitat.",
  negative_space: "Generate an image that uses negative space cleverly as a design element. The empty areas should work as hard as the filled areas, creating secondary images or meanings. Sophisticated visual puzzle that rewards attention.",
  neon_cyberpunk: "Generate an image drenched in neon lighting with cyberpunk atmosphere. Electric blues, hot pinks, and acid greens reflecting off wet surfaces. Urban night scenes with holographic advertisements and rain-slicked streets.",
  noir: "Generate an image in classic film noir style. High contrast black and white with deep shadows, Venetian blind patterns, smoke curling through shafts of light, and an atmosphere of mystery and tension.",
  oil_painting_classical: "Generate an image in the style of classical oil painting with rich, layered colors and visible brushwork. Employ chiaroscuro lighting, glazing techniques, and the color palettes of the old masters. Timeless and museum-worthy.",
  oil_painting_impressionist: "Generate an impressionist painting capturing fleeting light and atmosphere. Use visible brushstrokes, pure color placed side by side, and the play of natural light. Focus on the impression of a moment rather than precise detail.",
  paper_cut: "Generate an image in the style of layered paper cut art. Silhouettes creating depth through layering, visible paper edges, and the delicate craft of negative space carved from positive. Shadow boxes and dimensional paper sculpture. Fragile yet precise.",
  pastel_soft: "Generate a soft pastel artwork with luminous, chalky color. Layer colors optically, let the paper texture show through, and achieve that distinctive pastel glow. Soft edges and gentle transitions with bold color choices.",
  photorealistic: "Generate a photorealistic image with accurate lighting, natural textures, and realistic details. Pay attention to proper shadows, reflections, and material properties.",
  pixel_art: "Generate pixel art with a limited color palette and deliberate chunky pixels. Each pixel placed with intention, dithering for gradients, and the nostalgic charm of 8-bit and 16-bit era graphics. Readable silhouettes at low resolution.",
  pop_art: "Generate an image with pop art aesthetics. Bold colors, Ben-Day dots, and the elevation of commercial imagery to fine art. Mass media meets irony meets genuine appreciation. Flat, graphic, and immediately recognizable. The boundaries between high and low culture have been cheerfully demolished.",
  portrait_environmental: "Generate an environmental portrait that tells a story through the subject's surroundings. Balance the subject with their environment, using natural light and meaningful background elements that reveal character or profession.",
  portrait_studio: "Generate a professional studio portrait with controlled lighting, shallow depth of field, and flattering skin tones. Use classic three-point lighting with careful attention to catchlights and shadow placement.",
  powerpoint_peak: "Generate an image with maximum PowerPoint energy circa 2003. WordArt that bends reality. Clip art that asks questions about its own existence. Transition effects frozen mid-wipe. Background gradients that were chosen by committee. This slide has been loading for 20 years.",
  product_lifestyle: "Generate lifestyle product photography showing the product in real-world context. Aspirational settings, natural lighting, and staging that helps viewers imagine owning and using the product. Social media and advertising ready.",
  product_shot: "Generate clean product photography suitable for commercial use. Perfect lighting that shows form and materials, neutral or complementary background, and presentation that makes the product desirable. E-commerce and catalog ready.",
  prog_album: "Generate an image suitable for a 1970s progressive rock album cover. Cosmic landscapes, impossible architecture, and mystical symbolism. Roger Dean energy without the copyright issues. A visual representation of a 23-minute song about ancient civilizations and synthesizers.",
  propaganda: "Generate an image with the visual language of vintage propaganda posters. Bold colors, heroic poses, dramatic perspective from below, and simplified shapes for maximum impact. The message is unclear but the conviction is absolute. Someone is definitely marching somewhere.",
  psychedelic: "Generate an image with 1960s psychedelic art aesthetics. Melting forms, impossible colors, and patterns that seem to breathe. Paisley, fractal-like repetition, and the visual representation of expanded consciousness. Turn on, tune in, and generate art.",
  pulp_fiction: "Generate an image in the style of vintage pulp fiction covers. Dramatic scenes, lurid colors, and larger-than-life characters in mortal peril. Bold brush strokes and melodramatic compositions. Someone is always pointing a gun or swooning. The genre is unclear but the danger is immediate.",
  quality_boost: "Generate an image with maximum quality and refinement. Sharp where sharpness matters, smooth gradients without banding, accurate colors, and professional finish. Technical excellence in every aspect of the image.",
  real_estate: "Generate a real estate listing photo where the staging has achieved uncanny valley. Fruit bowls placed by algorithm. Throw pillows arranged to ward off evil. A kitchen where no meal has ever been prepared. Sunlight that seems contractually obligated. This could be your dream home if your dreams are suspicious.",
  renaissance_instagram: "Generate an image as if a Renaissance master were handed an iPhone and immediately became addicted to filters. Classical composition and chiaroscuro technique filtered through Valencia, with dramatic sfumato edges that might just be a vignette slider. Golden ratio meets golden hour preset.",
  retro_70s: "Generate an image with authentic 1970s aesthetic. Warm earth tones, avocado greens, harvest golds, and burnt oranges. Wood paneling textures, shag carpet energy, and the optimistic futurism of that era.",
  retro_80s: "Generate an image with bold 1980s aesthetic. Neon colors, chrome accents, geometric patterns, sunset gradients, and synthesizer vibes. VHS tracking artifacts optional. Everything the 80s thought the future would look like.",
  rim_light: "Generate an image with strong rim lighting that creates glowing edges and silhouettes. Backlight separates subject from background, creating halos and ethereal outlines. Dramatic and visually striking separation.",
  scifi_paperback: "Generate an image like a 1970s science fiction paperback cover. Alien landscapes, chrome spaceships, and people in jumpsuits looking toward distant horizons. The future as imagined before it arrived. Optimistic technology and cosmic wonder painted in oils.",
  screensaver_eternal: "Generate an image capturing the meditative essence of classic computer screensavers. Pipes that build toward nothing. Starfields promising journeys never taken. Maze walls that exist only to be forgotten. The screen has been saving for so long it has achieved enlightenment.",
  soft_diffused: "Generate an image with soft, diffused lighting like an overcast day or large softbox. Gentle shadows, smooth gradations, and flattering illumination. Peaceful and approachable with minimal harsh contrasts.",
  solarpunk: "Generate an image with solarpunk aesthetic. Optimistic visions of sustainable futures where technology and nature harmonize. Green architecture, renewable energy integrated beautifully, and communities thriving in balance with ecosystems. Hope made visible.",
  split_tone: "Generate an image with split toning applied. Warm colors in highlights, cool colors in shadows, and the sophisticated color grading that defines professional photography. The subtle technique that separates polished work from snapshots.",
  stained_glass: "Generate an image in the style of stained glass windows. Bold lead lines separating pieces of jewel-toned glass, light seeming to pass through and glow. Medieval cathedral energy meets graphic design clarity. Sacred geometry in colored light.",
  steampunk: "Generate an image with steampunk aesthetic. Victorian elegance meets brass-and-copper machinery. Gears visible whether or not they're functional, airships in amber skies, and the retro-future that steam power never delivered. Goggles optional but encouraged.",
  stock_photo_nightmare: "Generate the kind of stock photo that makes graphic designers weep. Forced smiles, inexplicable handshakes, people pointing at blank whiteboards with religious fervor, and thumbs-up poses that suggest a hostage situation. Aggressively inoffensive yet deeply unsettling.",
  street_candid: "Generate candid street photography that captures authentic moments of urban life. Look for decisive moments, interesting juxtapositions, and human stories unfolding naturally. Raw and unposed with strong composition.",
  studio_dramatic: "Generate an image with dramatic studio lighting. Strong contrast between light and shadow, carefully placed key light, and theatrical mood. The lighting tells as much story as the subject itself.",
  surrealist: "Generate a surrealist image drawn from the unconscious mind. Impossible juxtapositions, dreamlike logic, and objects that refuse to behave normally. Melting clocks optional but that energy is required. Reality is merely a suggestion. The id is in charge now.",
  textbook_diagram: "Generate an image like a textbook diagram that explains something simple with maximum complexity. Arrows pointing to other arrows. Labels in a font chosen for its inability to be read. A key that requires its own key. Educational content that raises more questions than it answers.",
  texture_focus: "Generate an image where texture is the star. Whether rough stone, smooth silk, weathered wood, or rusted metal, make the surface quality tactile and compelling. The viewer should almost feel the texture through the screen.",
  tilt_shift: "Generate an image with tilt-shift miniature effect. Sharp selective focus with dramatic blur falloff creates the illusion that real scenes are tiny models. High vantage point, saturated colors, and that uncanny sense of looking at a diorama of reality.",
  ukiyo_e: "Generate an image in the style of traditional Japanese ukiyo-e woodblock prints. Bold outlines, flat colors, and masterful use of negative space. Capture the essence of the floating world with careful attention to wave patterns, cloud formations, and the delicate balance of composition.",
  vaporwave_deadline: "Generate an image with vaporwave aesthetics but the anxiety of a missed deadline showing through. Greek statues but they look stressed. Palm trees that seem to be sweating. Windows 95 error messages rendered in soothing pink and teal. The S U N S E T is loading forever.",
  victorian_tech: "Generate an image as if a Victorian-era illustrator were asked to depict modern technology based on a confusing description. Smartphones rendered as brass contraptions with tiny people inside. Laptops that require coal. WiFi signals visualized as ghostly tendrils. Steam-powered everything.",
  vintage_film: "Generate an image with the aesthetic of vintage film photography. Film grain, slightly muted colors, subtle light leaks, and the organic imperfections of analog photography. Nostalgic without feeling artificially aged.",
  watercolor_botanical: "Generate a precise botanical watercolor illustration with scientific accuracy and artistic beauty. Render every vein, petal, and stem with careful observation. Combine the rigor of scientific illustration with the delicacy of fine watercolor technique.",
  watercolor_loose: "Generate a loose watercolor painting with flowing washes and happy accidents. Let colors bleed and bloom naturally, preserve white paper for highlights, and embrace the spontaneous nature of the medium. Fresh and luminous.",
  weirdcore: "Generate an image with weirdcore aesthetic. Low-quality imagery pushed into the uncanny, eye-straining colors, and compositions that feel fundamentally wrong. The visual language of corrupted memories and forgotten websites. Reality buffering incorrectly.",
  wes_anderson_fever: "Generate an image with obsessive symmetry, pastel color palettes, and deadpan whimsy cranked to pathological levels. Every object perfectly centered, every color meticulously coordinated, every surface flat and frontal. The visual equivalent of alphabetizing your sock drawer by thread count.",
  wireframe: "Generate an image showing 3D wireframe construction. Visible edge lines, hidden line removal optional, and the skeletal structure of three-dimensional forms. The blueprint before rendering, the architecture beneath surfaces. Technical beauty in pure geometry.",
  yearbook_90s: "Generate an image with 1990s school yearbook photo energy. Laser backgrounds in impossible colors. Hair that defies physics. Poses suggested by photographers who stopped caring decades ago. A gradient that transitions through several bad decisions. Frosted tips optional but encouraged.",
};

app.registerExtension({
  name: "Comfy.ZImageTextEncoder",

  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeData.name !== "ZImageTextEncoder") return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const ret = onNodeCreated
        ? onNodeCreated.apply(this, arguments)
        : undefined;

      const node = this;

      // Find widgets after a short delay to ensure they're initialized
      setTimeout(() => {
        const templateWidget = node.widgets?.find(
          (w) => w.name === "template_preset"
        );
        const systemPromptWidget = node.widgets?.find(
          (w) => w.name === "system_prompt"
        );

        if (!templateWidget || !systemPromptWidget) {
          console.log("ZImageTextEncoder: Missing widget", {
            templateWidget: !!templateWidget,
            systemPromptWidget: !!systemPromptWidget,
          });
          return;
        }

        // Store original callback
        const originalCallback = templateWidget.callback;

        // Override callback to update system_prompt field
        templateWidget.callback = function (value) {
          // Call original callback if exists
          if (originalCallback) {
            originalCallback.call(this, value);
          }

          // Update system_prompt based on preset
          if (Z_IMAGE_TEMPLATES[value] !== undefined) {
            systemPromptWidget.value = Z_IMAGE_TEMPLATES[value] || "";
          }

          // Mark node as needing update
          node.setDirtyCanvas(true, true);
        };

        // Trigger initial update if there's a value
        if (templateWidget.value && Z_IMAGE_TEMPLATES[templateWidget.value] !== undefined) {
          systemPromptWidget.value = Z_IMAGE_TEMPLATES[templateWidget.value] || "";
        }
      }, 10);

      return ret;
    };
  },
});
