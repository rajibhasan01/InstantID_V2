from instant_photo_maker.infer import generate_image

face_image_path = "/home/webrnd/Desktop/development/production/InstantID/InstantID_V2/images/razib1.jpg"
pose_image_path = None

prompt = "A photo of a pirates"
negative_prompt = "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry, deformed cat, deformed, nudity,naked, bikini, skimpy, scanty, bare skin, lingerie, swimsuit, exposed, see-through, photo, anthropomorphic cat, monochrome, pet collar, gun, weapon, blue, 3d, drones, drone, buildings in background, green"
output_count = 4
seed = 1104730247

for i in range(4):
    seed = seed + i
    image = generate_image(
                face_image_path=face_image_path,
                pose_image_path=pose_image_path,
                prompt=prompt,
                negative_prompt=negative_prompt,
                style_name="Vibrant Color",
                enhance_face_region=True,
                num_steps=30,
                identitynet_strength_ratio=0.80,
                adapter_strength_ratio=0.90,
                guidance_scale=5,
                seed=seed)
    image.save(f'output_{i}.jpg')